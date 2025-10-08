#!/usr/bin/python
#!-*-coding:utf8-*-


from .qwen2_vl_forward import patched_qwen2_vl_forward
from .qwen2_5_vl_forward import patched_qwen2_5_vl_forward

__all__ = ['patched_qwen2_vl_forward', 'patched_qwen2_5_vl_forward']
