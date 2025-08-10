"""LongBench v2 evaluation for long-context understanding and reasoning."""

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template

if TYPE_CHECKING:
    pass


class LongBenchEvaluator:
    """Evaluates language models on LongBench v2 long-context tasks."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize evaluator with model and tokenizer."""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        
        # Apply RoPE configuration if specified in eval_args
        if hasattr(self.eval_args, 'rope_scaling_type') and self.eval_args.rope_scaling_type:
            self._apply_rope_config()
        
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"
        
        # Configure chat template from tokenizer if available
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
            self.model_args.rope_scaling = rope_type
            print(f"Set model_args.rope_scaling = {rope_type}")
        elif rope_type in ["yarn", "longrope", "llama3"]:
            self.model_args.rope_scaling = "linear"
            
            rope_config = {"type": rope_type, "factor": scaling_factor}
            
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
                rope_config.update({
                    "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192, "rope_type": "llama3"
                })
            
            self.advanced_rope_config = rope_config
            print(f"Prepared advanced RoPE config: {rope_config}")
        else:
            print(f"Unknown RoPE type: {rope_type}, using model defaults")

    def _apply_advanced_rope_config(self) -> None:
        """Apply advanced RoPE configuration directly to the model after loading."""
        try:
            config = self.advanced_rope_config
            print(f"Applying advanced RoPE configuration to model: {config}")
            
            model_config = self.model.config
            if hasattr(model_config, 'rope_scaling'):
                model_config.rope_scaling = config
                print(f"Successfully overrode model rope_scaling: {model_config.rope_scaling}")
                
                if 'factor' in config and config['factor'] > 1:
                    original_max_pos = getattr(model_config, 'max_position_embeddings', 8192)
                    if hasattr(self.eval_args, 'longbench_max_length') and self.eval_args.longbench_max_length:
                        max_length = self.eval_args.longbench_max_length
                        if max_length > original_max_pos:
                            model_config.max_position_embeddings = max_length
                            print(f"Updated max_position_embeddings from {original_max_pos} to {max_length}")
            else:
                print("Model does not support rope_scaling configuration")
        except Exception as e:
            print(f"Failed to apply advanced RoPE config: {e}")
            print("Continuing with model's default RoPE configuration")

    def _configure_chat_template(self) -> None:
        """Configure chat template from tokenizer config if available."""
        try:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                print("Using tokenizer's built-in chat template")
                return
                
            tokenizer_config_path = os.path.join(self.model_args.model_name_or_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                
                if 'chat_template' in tokenizer_config:
                    self.tokenizer.chat_template = tokenizer_config['chat_template']
                    print("Loaded chat template from tokenizer_config.json")
                    return
        except Exception as e:
            print(f"Could not load chat template from tokenizer config: {e}")
            
        print(f"Using LlamaFactory template system (template: {self.data_args.template})")

    def _configure_generation_config(self) -> None:
        """Load and configure generation config from model."""
        try:
            if hasattr(self.model, 'generation_config') and self.model.generation_config:
                self.base_generation_config = self.model.generation_config
                print("Loaded generation config from model")
            else:
                self.base_generation_config = None
                print("No generation config found in model")
        except Exception as e:
            print(f"Could not load generation config: {e}")
            self.base_generation_config = None

    def _encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode messages using tokenizer chat template if available."""
        try:
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prompt_messages = [msg for msg in messages if msg["content"].strip()]
                encoded = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors=None
                )
                return encoded
        except Exception as e:
            print(f"Chat template encoding failed, falling back to LlamaFactory template: {e}")
        
        input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
        return input_ids

    @torch.inference_mode()
    def generate_response(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """Generate model response for given input."""
        try:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.eval_args.longbench_max_new_tokens,
                "temperature": self.eval_args.longbench_temperature,
                "do_sample": self.eval_args.longbench_temperature > 0,
                "top_p": self.eval_args.longbench_top_p,
                "top_k": self.eval_args.longbench_top_k,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            if self.base_generation_config:
                base_kwargs = {}
                if hasattr(self.base_generation_config, 'eos_token_id') and self.base_generation_config.eos_token_id:
                    base_kwargs['eos_token_id'] = self.base_generation_config.eos_token_id
                if hasattr(self.base_generation_config, 'bos_token_id') and self.base_generation_config.bos_token_id:
                    base_kwargs['bos_token_id'] = self.base_generation_config.bos_token_id
                
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

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer (A, B, C, D) from model response."""
        response = response.replace('*', '')
        
        patterns = [
            r'The correct answer is \(([A-D])\)',
            r'The correct answer is ([A-D])',
            r'answer is \(([A-D])\)',
            r'answer is ([A-D])',
            r'Answer: \(([A-D])\)',
            r'Answer: ([A-D])',
            r'\(([A-D])\)',
            r'^([A-D])$',
            r'^([A-D])\.',
            r'^([A-D])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Last resort: find any A, B, C, D in response
        letters = re.findall(r'\b([A-D])\b', response.upper())
        if letters:
            return letters[0]
        
        return None

    def truncate_context(self, context: str, max_length: int) -> str:
        """Truncate context to fit within max_length tokens."""
        input_ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(input_ids) <= max_length:
            return context
        
        # Use middle truncation to preserve beginning and end
        half_len = max_length // 2
        truncated_ids = input_ids[:half_len] + input_ids[-half_len:]
        return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def eval(self) -> None:
        """Run LongBench v2 evaluation."""
        print("Starting LongBench v2 evaluation...")
        
        # Load dataset
        try:
            dataset = load_dataset('zai-org/LongBench-v2', split='train')
            data_all = [{
                "_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"], "length": item["length"], "question": item["question"],
                "choice_A": item["choice_A"], "choice_B": item["choice_B"], 
                "choice_C": item["choice_C"], "choice_D": item["choice_D"],
                "answer": item["answer"], "context": item["context"]
            } for item in dataset]
        except Exception as e:
            print(f"Failed to load LongBench v2 dataset: {e}")
            return
        
        # Filter by domain/difficulty if specified
        if hasattr(self.eval_args, 'longbench_domains') and self.eval_args.longbench_domains:
            domains = self.eval_args.longbench_domains
            data_all = [item for item in data_all if item['domain'] in domains]
            print(f"Filtered to domains: {domains}, {len(data_all)} examples")
        
        if hasattr(self.eval_args, 'longbench_difficulty') and self.eval_args.longbench_difficulty:
            difficulty = self.eval_args.longbench_difficulty
            data_all = [item for item in data_all if item['difficulty'] == difficulty]
            print(f"Filtered to difficulty: {difficulty}, {len(data_all)} examples")
        
        # Limit number of examples if specified
        if hasattr(self.eval_args, 'longbench_max_examples') and self.eval_args.longbench_max_examples > 0:
            data_all = data_all[:self.eval_args.longbench_max_examples]
            print(f"Limited to {len(data_all)} examples")
        
        results = []
        correct_count = 0
        
        prompt_template = self.eval_args.longbench_prompt_template if hasattr(self.eval_args, 'longbench_prompt_template') else "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\nFormat your response as follows: \"The correct answer is (insert answer here)\"."
        
        for i, item in enumerate(tqdm(data_all, desc="Evaluating LongBench v2")):
            # Prepare context (truncate if needed)
            context = item['context']
            if hasattr(self.eval_args, 'longbench_max_context_length') and self.eval_args.longbench_max_context_length > 0:
                context = self.truncate_context(context, self.eval_args.longbench_max_context_length)
            
            # Format prompt
            prompt = prompt_template.format(
                context=context,
                question=item['question'],
                choice_A=item['choice_A'],
                choice_B=item['choice_B'],
                choice_C=item['choice_C'],
                choice_D=item['choice_D']
            )
            
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
            
            # Encode and generate
            try:
                input_ids = self._encode_messages(messages)
                input_tensor = torch.tensor([input_ids], device=self.model.device)
                attention_mask = torch.ones_like(input_tensor)
                
                if input_tensor.shape[1] > self.model.config.max_position_embeddings:
                    print(f"Skipping example {i}: input too long ({input_tensor.shape[1]} > {self.model.config.max_position_embeddings})")
                    response, pred_answer = "", None
                else:
                    response = self.generate_response(input_tensor, attention_mask)
                    pred_answer = self.extract_answer(response)
                
                is_correct = pred_answer == item['answer'] if pred_answer else False
                if is_correct:
                    correct_count += 1
                
                result = {
                    "example_id": item["_id"],
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                    "question": item["question"],
                    "correct_answer": item["answer"],
                    "predicted_answer": pred_answer,
                    "response": response,
                    "is_correct": is_correct,
                    "context_length": len(self.tokenizer.encode(context)),
                }
                
                # Save context if requested
                if hasattr(self.eval_args, 'longbench_save_context') and self.eval_args.longbench_save_context:
                    result["context"] = context[:1000]  # Save first 1000 chars
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
        
        self._save_results(results, correct_count, len(results))

    def _save_results(self, results: List[Dict], correct_count: int, total_count: int) -> None:
        """Save evaluation results and print summary."""
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # Group by domain and difficulty
        domain_stats = {}
        difficulty_stats = {}
        
        for result in results:
            domain = result['domain']
            difficulty = result['difficulty']
            
            if domain not in domain_stats:
                domain_stats[domain] = {'correct': 0, 'total': 0}
            domain_stats[domain]['correct'] += result['is_correct']
            domain_stats[domain]['total'] += 1
            
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
            difficulty_stats[difficulty]['correct'] += result['is_correct']
            difficulty_stats[difficulty]['total'] += 1
        
        # Calculate accuracies
        for domain in domain_stats:
            domain_stats[domain]['accuracy'] = domain_stats[domain]['correct'] / domain_stats[domain]['total']
        
        for difficulty in difficulty_stats:
            difficulty_stats[difficulty]['accuracy'] = difficulty_stats[difficulty]['correct'] / difficulty_stats[difficulty]['total']
        
        summary = {
            "overall": {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total_count
            },
            "by_domain": domain_stats,
            "by_difficulty": difficulty_stats
        }
        
        # Print summary
        print(f"\n=== LongBench v2 Evaluation Results ===")
        print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
        
        print(f"\nAccuracy by Domain:")
        for domain, stats in domain_stats.items():
            print(f"  {domain}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        print(f"\nAccuracy by Difficulty:")
        for difficulty, stats in difficulty_stats.items():
            print(f"  {difficulty}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        
        # Save results if save_dir specified
        if self.eval_args.save_dir:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)
            
            # Save detailed results
            with open(os.path.join(self.eval_args.save_dir, "longbench_detailed_results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save summary
            with open(os.path.join(self.eval_args.save_dir, "longbench_summary.json"), "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to {self.eval_args.save_dir}/")


def run_longbench_eval() -> None:
    """Entry point for LongBench v2 evaluation."""
    LongBenchEvaluator().eval()