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
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)

    @torch.inference_mode()
    def generate_response(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """Generate model response for given input."""
        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
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
            # Force regeneration to avoid cache issues
            builder.download_and_prepare(download_mode=self.eval_args.download_mode)
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
            
            input_ids, _ = self.template.encode_oneturn(
                tokenizer=self.tokenizer, messages=messages
            )
            input_tensor = torch.tensor([input_ids], device=self.model.device)
            attention_mask = torch.ones_like(input_tensor)
            
            if input_tensor.shape[1] > self.model.config.max_position_embeddings:
                response, score = "", 0.0
            else:
                response = self.generate_response(input_tensor, attention_mask)
                score = self.evaluate_needle_retrieval(response, example['needle'])
            
            scores.append(score)
            
            results.append({
                "example_id": i,
                "context_length": example['context_length_tokens'],
                "depth_percent": example['depth_percent'],
                "question": example['question'],
                "needle": example['needle'],
                "response": response,
                "score": score,
                "exact_match": 1.0 if score == 1.0 else 0.0
            })

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
        
        with open(os.path.join(self.eval_args.save_dir, "detailed_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(self.eval_args.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

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