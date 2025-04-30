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


def apply_sequence_parallel(model_args, full_determinism=False):
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

    return group_this
