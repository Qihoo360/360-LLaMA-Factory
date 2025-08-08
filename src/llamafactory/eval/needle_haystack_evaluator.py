"""Needle in haystack evaluation for long-context retrieval testing."""

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NeedleHaystackEvaluator:
    """Evaluates language models on needle-in-haystack retrieval tasks."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize evaluator with model and tokenizer."""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        
        # Apply RoPE configuration if specified in eval_args
        if hasattr(self.eval_args, 'rope_scaling_type') and self.eval_args.rope_scaling_type:
            self._apply_rope_config()
        
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"
        
        # Configure chat template from tokenizer if available, fallback to LlamaFactory template
        self._configure_chat_template()
        
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        
        # Apply advanced RoPE configuration if prepared
        if hasattr(self, 'advanced_rope_config'):
            self._apply_advanced_rope_config()
        
        # Load and configure generation config from model
        self._configure_generation_config()

    def _apply_rope_config(self) -> None:
        """Apply RoPE configuration from eval_args to model_args."""
        rope_type = self.eval_args.rope_scaling_type.lower()
        scaling_factor = self.eval_args.rope_scaling_factor or 2.0
        
        print(f"Applying RoPE configuration: {rope_type} (factor: {scaling_factor})")
        
        if rope_type in ["linear", "dynamic"]:
            # Use LlamaFactory's built-in RoPE scaling
            self.model_args.rope_scaling = rope_type
            print(f"Set model_args.rope_scaling = {rope_type}")
            
        elif rope_type in ["yarn", "longrope", "llama3"]:
            # For advanced RoPE types, we'll override the config after model loading
            # Set basic linear scaling first, then override with advanced config
            self.model_args.rope_scaling = "linear"
            
            # Store advanced RoPE config to apply after model loading
            rope_config = {
                "type": rope_type,
                "factor": scaling_factor
            }
            
            if rope_type == "yarn":
                if hasattr(self.eval_args, 'yarn_alpha') and self.eval_args.yarn_alpha:
                    rope_config["alpha"] = self.eval_args.yarn_alpha
                if hasattr(self.eval_args, 'yarn_beta') and self.eval_args.yarn_beta:
                    rope_config["beta"] = self.eval_args.yarn_beta
                    
            elif rope_type == "longrope":
                if hasattr(self.eval_args, 'longrope_short_factor') and self.eval_args.longrope_short_factor:
                    rope_config["short_factor"] = self.eval_args.longrope_short_factor
                if hasattr(self.eval_args, 'longrope_long_factor') and self.eval_args.longrope_long_factor:
                    rope_config["long_factor"] = self.eval_args.longrope_long_factor
                    
            elif rope_type == "llama3":
                # Llama3 RoPE has specific high/low frequency factors
                rope_config["high_freq_factor"] = 4.0
                rope_config["low_freq_factor"] = 1.0
                rope_config["original_max_position_embeddings"] = 8192
                rope_config["rope_type"] = "llama3"
            
            # Store for later application
            self.advanced_rope_config = rope_config
            print(f"Prepared advanced RoPE config: {rope_config}")
        
        else:
            print(f"Unknown RoPE type: {rope_type}, using model defaults")

    def _configure_chat_template(self) -> None:
        """Configure chat template from tokenizer config if available."""
        try:
            # Check if tokenizer already has a chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                print(f"✓ Using tokenizer's built-in chat template")
                return
                
            # Try to load chat template from tokenizer config
            tokenizer_config_path = os.path.join(self.model_args.model_name_or_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                import json
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                
                if 'chat_template' in tokenizer_config:
                    self.tokenizer.chat_template = tokenizer_config['chat_template']
                    print(f"✓ Loaded chat template from tokenizer_config.json")
                    return
                    
        except Exception as e:
            print(f"⚠ Could not load chat template from tokenizer config: {e}")
            
        print(f"ℹ Using LlamaFactory template system (template: {self.data_args.template})")

    def _apply_advanced_rope_config(self) -> None:
        """Apply advanced RoPE configuration directly to the model after loading."""
        try:
            if not hasattr(self, 'advanced_rope_config'):
                return
                
            config = self.advanced_rope_config
            print(f"Applying advanced RoPE configuration to model: {config}")
            
            # Get the model's config
            model_config = self.model.config
            
            # Apply the RoPE scaling configuration
            if hasattr(model_config, 'rope_scaling'):
                # Override the existing rope_scaling with our advanced config
                model_config.rope_scaling = config
                print(f"Successfully overrode model rope_scaling: {model_config.rope_scaling}")
                
                # Also update max_position_embeddings if we're scaling up
                if 'factor' in config and config['factor'] > 1:
                    original_max_pos = getattr(model_config, 'max_position_embeddings', 8192)
                    if hasattr(self.eval_args, 'needle_context_lengths') and self.eval_args.needle_context_lengths:
                        max_context = max(self.eval_args.needle_context_lengths)
                        if max_context > original_max_pos:
                            model_config.max_position_embeddings = max_context
                            print(f"Updated max_position_embeddings from {original_max_pos} to {max_context}")
            else:
                print("Model does not support rope_scaling configuration")
                
        except Exception as e:
            print(f"Failed to apply advanced RoPE config: {e}")
            print("Continuing with model's default RoPE configuration")

    def _configure_generation_config(self) -> None:
        """Load and configure generation config from model."""
        try:
            # Get the model's default generation config
            if hasattr(self.model, 'generation_config') and self.model.generation_config:
                self.base_generation_config = self.model.generation_config
                print(f"✓ Loaded generation config from model")
                
                # Print some key default settings
                if hasattr(self.base_generation_config, 'max_length'):
                    print(f"  - Default max_length: {self.base_generation_config.max_length}")
                if hasattr(self.base_generation_config, 'do_sample'):
                    print(f"  - Default do_sample: {self.base_generation_config.do_sample}")
                if hasattr(self.base_generation_config, 'temperature'):
                    print(f"  - Default temperature: {self.base_generation_config.temperature}")
            else:
                self.base_generation_config = None
                print(f"ℹ No generation config found in model, using manual configuration")
                
        except Exception as e:
            print(f"⚠ Could not load generation config: {e}")
            self.base_generation_config = None

    def _encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode messages using tokenizer chat template if available, otherwise LlamaFactory template."""
        try:
            # Try tokenizer's chat template first
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                # Remove empty assistant message for prompt-only encoding
                prompt_messages = [msg for msg in messages if msg["content"].strip()]
                
                encoded = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None
                )
                return encoded
        except Exception as e:
            print(f"⚠ Chat template encoding failed, falling back to LlamaFactory template: {e}")
        
        # Fallback to LlamaFactory template
        input_ids, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, messages=messages
        )
        return input_ids

    @torch.inference_mode()
    def generate_response(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """Generate model response for given input."""
        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            # Start with base generation config if available, then override with eval_args
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.eval_args.needle_generation_max_tokens,
                "temperature": self.eval_args.needle_generation_temperature,
                "do_sample": self.eval_args.needle_generation_temperature > 0,
                "top_p": self.eval_args.needle_generation_top_p,
                "top_k": self.eval_args.needle_generation_top_k,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Merge with model's generation config if available
            if self.base_generation_config:
                # Start with model defaults
                base_kwargs = {}
                if hasattr(self.base_generation_config, 'eos_token_id') and self.base_generation_config.eos_token_id:
                    base_kwargs['eos_token_id'] = self.base_generation_config.eos_token_id
                if hasattr(self.base_generation_config, 'bos_token_id') and self.base_generation_config.bos_token_id:
                    base_kwargs['bos_token_id'] = self.base_generation_config.bos_token_id
                if hasattr(self.base_generation_config, 'repetition_penalty') and self.base_generation_config.repetition_penalty:
                    base_kwargs['repetition_penalty'] = self.base_generation_config.repetition_penalty
                
                # Only use model defaults if eval_args don't override
                for key, value in base_kwargs.items():
                    if key not in generation_kwargs:
                        generation_kwargs[key] = value
            
            outputs = self.model.generate(**generation_kwargs)
            
            input_length = input_ids.shape[1]
            if outputs.shape[1] > input_length:
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                return response.strip()
            return ""
                
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def evaluate_needle_retrieval(self, response: str, needle: str) -> float:
        """Evaluate needle retrieval accuracy."""
        needle_keywords = self._extract_keywords(needle)
        response_lower = response.lower()
        
        if not needle_keywords:
            return 0.0
        
        if needle.lower().strip() in response_lower:
            return 1.0
        
        matches = sum(1 for keyword in needle_keywords 
                     if keyword.lower() in response_lower)
        return matches / len(needle_keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
            'in', 'with', 'to', 'for', 'of', 'as', 'by'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]

    def eval(self) -> None:
        """Run needle haystack evaluation."""
        task_parts = self.eval_args.task.split("_")
        eval_task = "_".join(task_parts[:2])
        
        # Build config for needle haystack with custom parameters
        config_kwargs = {}
        if hasattr(self.eval_args, 'needle_context_lengths') and self.eval_args.needle_context_lengths:
            config_kwargs['context_lengths'] = self.eval_args.needle_context_lengths
        if hasattr(self.eval_args, 'needle_depth_percents') and self.eval_args.needle_depth_percents:
            config_kwargs['document_depth_percents'] = self.eval_args.needle_depth_percents
        if hasattr(self.eval_args, 'needle_text') and self.eval_args.needle_text:
            config_kwargs['needle'] = self.eval_args.needle_text
        if hasattr(self.eval_args, 'needle_question') and self.eval_args.needle_question:
            config_kwargs['retrieval_question'] = self.eval_args.needle_question
        if hasattr(self.eval_args, 'needle_haystack_data_source') and self.eval_args.needle_haystack_data_source:
            config_kwargs['data_source'] = self.eval_args.needle_haystack_data_source
        if hasattr(self.eval_args, 'needle_haystack_data_dir') and self.eval_args.needle_haystack_data_dir:
            config_kwargs['data_dir'] = self.eval_args.needle_haystack_data_dir
            
        # Import the config class and load dataset
        dataset_path = os.path.join(self.eval_args.task_dir, eval_task)
        if config_kwargs and eval_task == "needle_haystack":
            # For custom config, we need to use the dataset builder directly
            import sys
            sys.path.append(dataset_path)
            from needle_haystack_proper import NeedleHaystackProper, NeedleHaystackProperConfig
            
            config = NeedleHaystackProperConfig(name="needle_haystack_proper", **config_kwargs)
            builder = NeedleHaystackProper()
            builder.config = config
            
            # Clear cache if there's a mismatch error
            try:
                builder.download_and_prepare(download_mode=self.eval_args.download_mode)
            except Exception as e:
                if "NonMatchingSplitsSizeError" in str(e):
                    print("Detected cache mismatch. Clearing cache and regenerating dataset...")
                    # Force regeneration by using FORCE_REDOWNLOAD mode
                    from datasets import DownloadMode
                    builder.download_and_prepare(download_mode=DownloadMode.FORCE_REDOWNLOAD)
                else:
                    raise e
            
            dataset = builder.as_dataset(split="test")
            dataset = {"test": dataset}
        else:
            dataset = load_dataset(
                path=dataset_path,
                name=eval_task,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=True,
            )

        results, scores = [], []
        
        for i, example in enumerate(tqdm(dataset["test"], desc="Processing examples")):
            messages = [
                {
                    "role": "user", 
                    "content": f"Context: {example['context']}\n\nQuestion: {example['question']}"
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
            
            # Try to use tokenizer's chat template if available, otherwise use LlamaFactory template
            input_ids = self._encode_messages(messages)
            input_tensor = torch.tensor([input_ids], device=self.model.device)
            attention_mask = torch.ones_like(input_tensor)
            
            if input_tensor.shape[1] > self.model.config.max_position_embeddings:
                response, score = "", 0.0
            else:
                response = self.generate_response(input_tensor, attention_mask)
                score = self.evaluate_needle_retrieval(response, example['needle'])
            
            scores.append(score)
            
            result_dict = {
                "example_id": i,
                "context_length": example['context_length_tokens'],
                "depth_percent": example['depth_percent'],
                "question": example['question'],
                "needle": example['needle'],
                "response": response,
                "score": score,
                "exact_match": 1.0 if score == 1.0 else 0.0
            }
            
            # Save input prompt and token counts if requested
            if self.eval_args.needle_save_inputs_outputs:
                result_dict["input_prompt"] = f"Context: {example['context']}\n\nQuestion: {example['question']}"
                result_dict["input_tokens"] = input_tensor.shape[1]
                result_dict["output_tokens"] = len(self.tokenizer.encode(response)) if response else 0
            
            results.append(result_dict)

        self._save_results(results, scores)
        if self.eval_args.save_dir:
            self._generate_visualizations(results)

    def _save_results(self, results: List[Dict], scores: List[float]) -> None:
        """Save evaluation results and print summary."""
        avg_score = np.mean(scores)
        exact_match_rate = np.mean([r["exact_match"] for r in results])
        
        context_lengths = sorted(set(r["context_length"] for r in results))
        depth_percents = sorted(set(r["depth_percent"] for r in results))
        
        summary = {
            "overall": {
                "average_score": float(avg_score),
                "exact_match_rate": float(exact_match_rate),
                "total_examples": len(results)
            },
            "by_context_length": self._group_by_field(results, "context_length"),
            "by_depth_percent": self._group_by_field(results, "depth_percent")
        }

        self._print_summary(avg_score, exact_match_rate, len(results), 
                           context_lengths, depth_percents, summary)
        
        if self.eval_args.save_dir:
            self._save_files(results, summary)

    def _group_by_field(self, results: List[Dict], field: str) -> Dict:
        """Group results by specified field."""
        grouped = {}
        for value in set(r[field] for r in results):
            subset = [r for r in results if r[field] == value]
            subset_scores = [r["score"] for r in subset]
            grouped[str(value)] = {
                "average_score": float(np.mean(subset_scores)),
                "exact_match_rate": float(np.mean([r["exact_match"] for r in subset])),
                "count": len(subset)
            }
        return grouped

    def _print_summary(self, avg_score: float, exact_match_rate: float, 
                      total: int, context_lengths: List[int], 
                      depth_percents: List[float], summary: Dict) -> None:
        """Print evaluation summary."""
        print(f"\n=== Needle in Haystack Evaluation Results ===")
        print(f"Overall Average Score: {avg_score:.3f}")
        print(f"Exact Match Rate: {exact_match_rate:.3f}")
        print(f"Total Examples: {total}")
        
        print(f"\nScores by Context Length:")
        for length in context_lengths:
            score = summary["by_context_length"][str(length)]["average_score"]
            print(f"  {length:,} tokens: {score:.3f}")
        
        print(f"\nScores by Needle Depth:")
        for depth in depth_percents:
            score = summary["by_depth_percent"][str(depth)]["average_score"]
            print(f"  {depth}%: {score:.3f}")

    def _save_files(self, results: List[Dict], summary: Dict) -> None:
        """Save results to files."""
        os.makedirs(self.eval_args.save_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(self.eval_args.save_dir, "detailed_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        with open(os.path.join(self.eval_args.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save inputs and outputs separately if requested
        if self.eval_args.needle_save_inputs_outputs:
            inputs_outputs = []
            for r in results:
                if "input_prompt" in r:
                    inputs_outputs.append({
                        "example_id": r["example_id"],
                        "context_length": r["context_length"],
                        "depth_percent": r["depth_percent"],
                        "input_prompt": r["input_prompt"],
                        "response": r["response"],
                        "input_tokens": r.get("input_tokens", 0),
                        "output_tokens": r.get("output_tokens", 0),
                        "score": r["score"]
                    })
            
            if inputs_outputs:
                with open(os.path.join(self.eval_args.save_dir, "inputs_outputs.json"), "w") as f:
                    json.dump(inputs_outputs, f, indent=2)
                print(f"\nSaved input prompts and outputs to {os.path.join(self.eval_args.save_dir, 'inputs_outputs.json')}")

    def _generate_visualizations(self, results: List[Dict]) -> None:
        """Generate performance visualizations."""
        if not HAS_MATPLOTLIB:
            return
        
        plt.switch_backend('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Needle in Haystack Evaluation Results', fontsize=16, fontweight='bold')
        
        self._plot_by_context_length(results, ax1)
        self._plot_by_needle_position(results, ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_args.save_dir, "needle_haystack_performance.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self._generate_heatmap(results)

    def _plot_by_context_length(self, results: List[Dict], ax) -> None:
        """Plot performance by context length."""
        context_lengths = sorted(set(r["context_length"] for r in results))
        avg_scores, exact_match_rates = [], []
        
        for length in context_lengths:
            subset = [r for r in results if r["context_length"] == length]
            avg_scores.append(np.mean([r["score"] for r in subset]))
            exact_match_rates.append(np.mean([r["exact_match"] for r in subset]))
        
        x = np.arange(len(context_lengths))
        width = 0.35
        
        ax.bar(x - width/2, avg_scores, width, label='Average Score', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, exact_match_rates, width, label='Exact Match Rate', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Context Length (tokens)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance by Context Length')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l:,}' for l in context_lengths], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    def _plot_by_needle_position(self, results: List[Dict], ax) -> None:
        """Plot performance by needle position."""
        depth_percents = sorted(set(r["depth_percent"] for r in results))
        avg_scores, exact_match_rates = [], []
        
        for depth in depth_percents:
            subset = [r for r in results if r["depth_percent"] == depth]
            avg_scores.append(np.mean([r["score"] for r in subset]))
            exact_match_rates.append(np.mean([r["exact_match"] for r in subset]))
        
        x = np.arange(len(depth_percents))
        width = 0.35
        
        ax.bar(x - width/2, avg_scores, width, label='Average Score', color='lightgreen', alpha=0.8)
        ax.bar(x + width/2, exact_match_rates, width, label='Exact Match Rate', color='orange', alpha=0.8)
        
        ax.set_xlabel('Needle Position (% depth)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance by Needle Position')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(d)}%' for d in depth_percents])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    def _generate_heatmap(self, results: List[Dict]) -> None:
        """Generate performance heatmap."""
        try:
            context_lengths = sorted(set(r["context_length"] for r in results))
            depth_percents = sorted(set(r["depth_percent"] for r in results))
            
            matrix = np.full((len(depth_percents), len(context_lengths)), np.nan)
            
            for i, depth in enumerate(depth_percents):
                for j, length in enumerate(context_lengths):
                    subset = [r for r in results 
                             if r["depth_percent"] == depth and r["context_length"] == length]
                    if subset:
                        matrix[i, j] = np.mean([r["score"] for r in subset])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            ax.set_xticks(range(len(context_lengths)))
            ax.set_xticklabels([f'{l:,}' for l in context_lengths], rotation=45)
            ax.set_yticks(range(len(depth_percents)))
            ax.set_yticklabels([f'{int(d)}%' for d in depth_percents])
            
            ax.set_xlabel('Context Length (tokens)')
            ax.set_ylabel('Needle Position (% depth)')
            ax.set_title('Performance Heatmap')
            
            plt.colorbar(im, ax=ax, label='Performance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.eval_args.save_dir, "needle_haystack_heatmap.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception:
            pass


def run_needle_haystack_eval() -> None:
    """Entry point for needle haystack evaluation."""
    NeedleHaystackEvaluator().eval()