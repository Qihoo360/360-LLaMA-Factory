# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial

import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from ring_flash_attn import zigzag_ring_flash_attn_func
from yunchang import set_seq_parallel_pg, EXTRACT_FUNC_DICT
from yunchang.kernels import AttnType
from .usp import LongContextAttention
from .ulysses import UlyssesAttention

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

def new_flash_attn_forward_usp(
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
    group_ulysses=None,
    group_ring=None,
    mode="usp",
    **kwargs,
):
    if mode == "usp":
        dist_attn = LongContextAttention(sequence_parallel_ulysses_group=group_ulysses, sequence_parallel_ring_group=group_ring, ring_impl_type="zigzag", attn_type=AttnType.FA)
        attn_output = dist_attn(query_states, key_states, value_states, deterministic=deterministic, dropout_p=dropout, causal=is_causal) # reset query_length to the real q_len before sp, Special settings for ulysses
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


def init_sp_usp(
    sp_ulysses_degree, sp_ring_degree, use_ulysses_low=True
):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert (
        world_size % sp_degree == 0
    ), f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"
    rank = dist.get_rank()

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    )
                )
                group = dist.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

            for i in range(num_ring_pgs):
                ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
                group = dist.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group
    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(
                    range(
                        i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset
                    )
                )
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(i + offset, sp_degree + offset, num_ulysses_pgs)
                )
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    return ulyssess_pg, ring_pg

def apply_sequence_parallel(model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    if model_args.sequence_parallel_mode == "usp":
        group_ulysses, group_ring = init_sp_usp(model_args.sequence_parallel_ulysses_degree, model_args.sequence_parallel_ring_degree)
    else:
        group_this = init_sp_group(model_args.sequence_parallel_size)
    original_attn = transformers.modeling_flash_attention_utils._flash_attention_forward

    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        elif model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism, attn_fn=original_attn, sequence_parallel_size=model_args.sequence_parallel_size)
        elif model_args.sequence_parallel_mode == "usp":
            new_flash_attention_forward = partial(new_flash_attn_forward_usp, group_ulysses=group_ulysses, group_ring=group_ring, mode=model_args.sequence_parallel_mode, deterministic=full_determinism, sequence_parallel_size=model_args.sequence_parallel_size)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/Qihoo360/360-llama-factory"
        )
    if model_args.sequence_parallel_mode == "usp":
        return group_ulysses, group_ring
    else:
        return group_this
