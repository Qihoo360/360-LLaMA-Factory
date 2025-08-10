# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    """

    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task_dir: str = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    lang: Literal["en", "zh"] = field(
        default="en",
        metadata={"help": "Language used at evaluation."},
    )
    n_shot: int = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."},
    )
    download_mode: DownloadMode = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets."},
    )
    # Needle haystack specific parameters
    needle_context_lengths: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Context lengths in tokens for needle haystack evaluation. Default: [250, 500, 1000, 2000]"},
    )
    needle_depth_percents: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Needle depth percentages for needle haystack evaluation. Default: [0, 25, 50, 75, 100]"},
    )
    needle_text: Optional[str] = field(
        default=None,
        metadata={"help": "Custom needle text for needle haystack evaluation. Default: 'The secret key is 42 alpha bravo.'"},
    )
    needle_question: Optional[str] = field(
        default=None,
        metadata={"help": "Custom retrieval question for needle haystack evaluation. Default: 'What is the secret key?'"},
    )
    needle_haystack_data_source: Optional[str] = field(
        default=None,
        metadata={"help": "Data source for needle haystack background text. Options: 'custom', 'paulgraham', 'directory'. Default: 'custom'"},
    )
    needle_haystack_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory path for custom haystack text files (when data_source='directory'). Default: evaluation/needle_haystack/data/PaulGrahamEssays"},
    )
    needle_save_inputs_outputs: bool = field(
        default=False,
        metadata={"help": "Save input prompts and model outputs for needle haystack evaluation"},
    )
    needle_generation_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for generation in needle haystack (0.0 = deterministic)"},
    )
    needle_generation_top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p for generation in needle haystack"},
    )
    needle_generation_top_k: int = field(
        default=50,
        metadata={"help": "Top-k for generation in needle haystack"},
    )
    needle_generation_max_tokens: int = field(
        default=50,
        metadata={"help": "Maximum new tokens for generation in needle haystack"},
    )
    # RoPE and context length evaluation parameters
    rope_scaling_type: Optional[str] = field(
        default=None,
        metadata={"help": "RoPE scaling type: linear, dynamic, yarn, longrope, llama3"},
    )
    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={"help": "RoPE scaling factor for extending context length"},
    )
    yarn_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "YARN RoPE alpha parameter"},
    )
    yarn_beta: Optional[float] = field(
        default=None,
        metadata={"help": "YARN RoPE beta parameter"},
    )
    longrope_short_factor: Optional[List[float]] = field(
        default=None,
        metadata={"help": "LongRoPE short factor list"},
    )
    longrope_long_factor: Optional[List[float]] = field(
        default=None,
        metadata={"help": "LongRoPE long factor list"},
    )
    # LongBench v2 evaluation parameters
    longbench_max_examples: int = field(
        default=0,
        metadata={"help": "Maximum number of LongBench examples to evaluate (0 = all)"},
    )
    longbench_domains: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LongBench domains to evaluate (e.g., ['single_document_qa', 'multi_document_qa'])"},
    )
    longbench_difficulty: Optional[str] = field(
        default=None,
        metadata={"help": "LongBench difficulty level: 'easy' or 'hard'"},
    )
    longbench_max_context_length: int = field(
        default=0,
        metadata={"help": "Maximum context length in tokens (0 = no truncation)"},
    )
    longbench_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum model length for LongBench evaluation"},
    )
    longbench_save_context: bool = field(
        default=False,
        metadata={"help": "Save context in results (first 1000 chars)"},
    )
    longbench_prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Custom prompt template for LongBench evaluation"},
    )
    longbench_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for LongBench generation"},
    )
    longbench_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for LongBench generation"},
    )
    longbench_top_k: int = field(
        default=1,
        metadata={"help": "Top-k for LongBench generation"},
    )
    longbench_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum new tokens for LongBench generation"},
    )
    longbench_mode: Optional[str] = field(
        default=None,
        metadata={"help": "LongBench evaluation mode: 'direct', 'vllm', 'official' (default: auto-detect)"},
    )

    def __post_init__(self):
        if self.save_dir is not None and os.path.exists(self.save_dir):
            raise ValueError("`save_dir` already exists, use another one.")
