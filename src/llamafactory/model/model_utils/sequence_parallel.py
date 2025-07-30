# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial

import torch.distributed as dist
import transformers
# import transformers.modeling_flash_attention_utils
from ring_flash_attn import zigzag_ring_flash_attn_func
from .ulysses import UlyssesAttention
from ...extras.packages import is_transformers_version_greater_than

from torch.nn import CrossEntropyLoss
from typing import Union, Tuple, List, Optional
import torch

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="zigzag-ring",
    attn_fn=None,
    **kwargs,
):
    if mode == "zigzag-ring":
        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(query_states, key_states, value_states, attention_mask, query_length=q_len * sequence_parallel_size, deterministic=deterministic, dropout_p=dropout, causal=is_causal) # reset query_length to the real q_len before sp, Special settings for ulysses
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(model_args, config, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)
    original_attn = transformers.modeling_flash_attention_utils._flash_attention_forward

    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        elif model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism, attn_fn=original_attn, sequence_parallel_size=model_args.sequence_parallel_size)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward

        # AttentionInterface for qwen3 and newer models
        if is_transformers_version_greater_than("4.51.0"):
            from transformers import AttentionInterface

            # modified from integrations/flash_attention.py
            from typing import Optional, Tuple

            import torch

            from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask


            _use_top_left_mask = flash_attn_supports_top_left_mask()


            def sequence_parallel_attention(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                dropout: float = 0.0,
                scaling: Optional[float] = None,
                sliding_window: Optional[int] = None,
                softcap: Optional[float] = None,
                **kwargs,
            ) -> Tuple[torch.Tensor, None]:
                # This is before the transpose
                seq_len = query.shape[2]

                # FA2 uses non-transposed inputs
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                # In PEFT, usually we cast the layer norms in float32 for training stability reasons
                # therefore the input hidden states gets silently casted in float32. Hence, we need
                # cast them back in the correct dtype just to be sure everything works as expected.
                # This might slowdown training & inference so it is recommended to not cast the LayerNorms
                # in fp32. (usually our RMSNorm modules handle it correctly)
                target_dtype = None
                if query.dtype == torch.float32:
                    if torch.is_autocast_enabled():
                        target_dtype = torch.get_autocast_gpu_dtype()
                    # Handle the case where the model is quantized
                    elif hasattr(module.config, "_pre_quantization_dtype"):
                        target_dtype = module.config._pre_quantization_dtype
                    else:
                        target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

                # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
                kwargs.pop("is_causal", None)

                attn_output = new_flash_attention_forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    q_len=seq_len,
                    is_causal=module.is_causal,
                    dropout=dropout,
                    softmax_scale=scaling,
                    sliding_window=sliding_window,
                    softcap=softcap,
                    use_top_left_mask=_use_top_left_mask,
                    target_dtype=target_dtype,
                    **kwargs,
                )

                return attn_output, None


            AttentionInterface.register("sequence_parallel_attention", sequence_parallel_attention)

    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/Qihoo360/360-llama-factory"
        )

    try:
        model_type = getattr(config, "model_type", None)
        if model_type == "qwen2_vl":
            import transformers.models.qwen2_vl.modeling_qwen2_vl as qwen_module
            _original_qwen_forward = qwen_module.Qwen2_VLForConditionalGeneration.forward

            def patched_qwen2_vl_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
            ):
            
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
                if inputs_embeds is None:
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    if pixel_values is not None:
                        pixel_values = pixel_values.type(self.visual.get_dtype())
                        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                        n_image_features = image_embeds.shape[0]
                        if n_image_tokens != n_image_features:
                            raise ValueError(
                                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                            )
                        image_mask = (
                            (input_ids == self.config.image_token_id)
                            .unsqueeze(-1)
                            .expand_as(inputs_embeds)
                            .to(inputs_embeds.device)
                        )
                        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
                    if pixel_values_videos is not None:
                        pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                        n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                        n_video_features = video_embeds.shape[0]
                        if n_video_tokens != n_video_features:
                            raise ValueError(
                                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                            )
                        video_mask = (
                            (input_ids == self.config.video_token_id)
                            .unsqueeze(-1)
                            .expand_as(inputs_embeds)
                            .to(inputs_embeds.device)
                        )
                        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(inputs_embeds.device)
            
                # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
                if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                    # calculate RoPE index once per generation in the pre-fill stage only
                    if (
                        (cache_position is not None and cache_position[0] == 0)
                        or self.rope_deltas is None
                        or (past_key_values is None or past_key_values.get_seq_length() == 0)
                    ):
                        position_ids, rope_deltas = self.get_rope_index(
                            input_ids, image_grid_thw, video_grid_thw, attention_mask
                        )
                        self.rope_deltas = rope_deltas
                    # then use the prev pre-calculated rope-deltas to get the correct position ids
                    else:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if cache_position is not None:  # otherwise `deltas` is an int `0`
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                            delta = delta.to(position_ids.device)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            
                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
            
                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
            
                loss = None
                if labels is not None:
                    # Upcast to float if we need to compute the loss to avoid potential precision issues
                    logits = logits.float()
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
            
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
            
                return qwen_module.Qwen2VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=self.rope_deltas,
                )
            

            qwen_module.Qwen2_VLForConditionalGeneration.forward = patched_qwen2_vl_forward
        elif model_type == "qwen2_5_vl":
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_module
            _original_qwen_forward = qwen_module.Qwen2_5_VLForConditionalGeneration.forward

            def patched_qwen2_5_vl_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                image_position_maps: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,
            ):
            
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
                if inputs_embeds is None:
                    inputs_embeds = self.model.embed_tokens(input_ids)

                    has_images_global = False
                    if pixel_values is not None:
                        has_images_local = torch.tensor(1, device=input_ids.device)
                    else:
                        has_images_local = torch.tensor(0, device=input_ids.device)
                    torch.distributed.all_reduce(has_images_local, op=torch.distributed.ReduceOp.MAX)
                    has_images_global = has_images_local.item() > 0

                    if has_images_global:
                        if pixel_values is not None:
                            if image_position_maps is not None:
                                pixel_values = pixel_values.type(self.visual.dtype)
                                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                
                                image_token_mask = (input_ids == self.config.image_token_id)
                                position_values = image_position_maps[image_token_mask]
                                reordered_image_embeds = image_embeds[position_values]
                                
                                image_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
                                reordered_image_embeds = reordered_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                                inputs_embeds = inputs_embeds.masked_scatter(image_mask, reordered_image_embeds)
                            else:
                                pixel_values = pixel_values.type(self.visual.dtype)
                                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                                n_image_features = image_embeds.shape[0]
                                if n_image_tokens != n_image_features:
                                    raise ValueError(
                                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                                    )
                
                                mask = input_ids == self.config.image_token_id
                                mask_unsqueezed = mask.unsqueeze(-1)
                                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                                image_mask = mask_expanded.to(inputs_embeds.device)
                
                                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                        else:
                            with torch.no_grad():
                                dummy_pixel_values = torch.zeros((4, 1176), device=input_ids.device, dtype=self.visual.dtype)
                                dummy_grid_thw = torch.tensor([[1, 2, 2]], device=input_ids.device)
                                _ = self.visual(dummy_pixel_values, grid_thw=dummy_grid_thw)
            
                    if pixel_values_videos is not None:
                        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                        n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                        n_video_features = video_embeds.shape[0]
                        if n_video_tokens != n_video_features:
                            raise ValueError(
                                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                            )
            
                        mask = input_ids == self.config.video_token_id
                        mask_unsqueezed = mask.unsqueeze(-1)
                        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                        video_mask = mask_expanded.to(inputs_embeds.device)
            
                        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(inputs_embeds.device)
            
                # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
                if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                    # calculate RoPE index once per generation in the pre-fill stage only
                    if (
                        (cache_position is not None and cache_position[0] == 0)
                        or self.rope_deltas is None
                        or (past_key_values is None or past_key_values.get_seq_length() == 0)
                    ):
                        position_ids, rope_deltas = self.get_rope_index(
                            input_ids,
                            image_grid_thw,
                            video_grid_thw,
                            second_per_grid_ts,
                            attention_mask,
                        )
                        self.rope_deltas = rope_deltas
                    # then use the prev pre-calculated rope-deltas to get the correct position ids
                    else:
                        batch_size, seq_length, _ = inputs_embeds.shape
                        delta = (
                            (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                            if cache_position is not None
                            else 0
                        )
                        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                        if cache_position is not None:  # otherwise `deltas` is an int `0`
                            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                        position_ids = position_ids.add(delta)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            
                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
            
                hidden_states = outputs[0]
                logits = self.lm_head(hidden_states)
            
                loss = None
                if labels is not None:
                    # Upcast to float if we need to compute the loss to avoid potential precision issues
                    logits = logits.float()
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
            
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
            
                return qwen_module.Qwen2_5_VLCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=self.rope_deltas,
                )


            qwen_module.Qwen2_5_VLForConditionalGeneration.forward = patched_qwen2_5_vl_forward
        else:
            print("Sequence parallelism is currently not supported for other multi-modal models. Please modify the corresponding model's forward function.")
    except:
        raise "Failed to patch multi-modal model"


    return group_this
