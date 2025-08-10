"""Direct LongBench v2 evaluation without server requirement."""

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class LongBenchV2DirectEvaluator:
    """Direct evaluation on LongBench v2 using loaded model."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize evaluator with model and tokenizer."""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        
        # Load tokenizer and model
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # For batch generation
        
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        
        # Set model to eval mode
        self.model.eval()
        
        # Load prompt template from official repo if available
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the official prompt template."""
        template_path = Path("third_party/LongBench/prompts/0shot.txt")
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Default template if official one not found
            return """Given the document below, answer the following question.

Document:
$DOC$

Question: $Q$

Choose the correct answer from the options below:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

The correct answer is ("""

    def truncate_context(self, text: str, max_length: int) -> str:
        """Truncate context to fit within max length using middle truncation."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return text
        
        # Middle truncation: keep beginning and end
        half_length = max_length // 2
        truncated_tokens = tokens[:half_length] + tokens[-half_length:]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format the prompt for the model."""
        # Get max context length
        max_context_length = getattr(self.eval_args, 'longbench_max_context_length', 32000)
        
        # Truncate context if needed
        context = self.truncate_context(item['context'], max_context_length - 1000)  # Reserve space for prompt
        
        # Format prompt using template
        prompt = self.prompt_template.replace('$DOC$', context.strip())
        prompt = prompt.replace('$Q$', item['question'].strip())
        prompt = prompt.replace('$C_A$', item['choice_A'].strip())
        prompt = prompt.replace('$C_B$', item['choice_B'].strip())
        prompt = prompt.replace('$C_C$', item['choice_C'].strip())
        prompt = prompt.replace('$C_D$', item['choice_D'].strip())
        
        return prompt
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from model response."""
        response = response.strip()
        
        # Try to match answer patterns
        patterns = [
            r'^([A-D])',  # Just the letter at the start
            r'answer is \(?([A-D])\)?',  # "answer is (A)" or "answer is A"
            r'correct answer is \(?([A-D])\)?',  # "correct answer is (A)"
            r'\(?([A-D])\)',  # Just "(A)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, check if response starts with A, B, C, or D
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
        
        return None
    
    @torch.no_grad()
    def generate_answer(self, prompt: str) -> str:
        """Generate answer from the model."""
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": getattr(self.eval_args, 'longbench_max_new_tokens', 128),
            "temperature": getattr(self.eval_args, 'longbench_temperature', 0.1),
            "top_p": getattr(self.eval_args, 'longbench_top_p', 1.0),
            "top_k": getattr(self.eval_args, 'longbench_top_k', 1),
            "do_sample": getattr(self.eval_args, 'longbench_temperature', 0.1) > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def eval(self) -> None:
        """Run LongBench v2 evaluation."""
        print("Starting LongBench v2 direct evaluation...")
        
        # Load dataset
        try:
            # Check if task_dir is specified for local dataset
            if hasattr(self.eval_args, 'task_dir') and self.eval_args.task_dir:
                dataset_path = self.eval_args.task_dir
                if os.path.exists(dataset_path):
                    print(f"Loading LongBench v2 dataset from local path: {dataset_path}")
                    # Try loading as JSON file
                    if dataset_path.endswith('.json'):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            data_all = json.load(f)
                    else:
                        dataset = load_dataset(dataset_path, split='train')
                        data_all = list(dataset)
                else:
                    print(f"Local path not found, trying as HuggingFace dataset ID: {dataset_path}")
                    dataset = load_dataset(dataset_path, split='train')
                    data_all = list(dataset)
            else:
                # Default to HuggingFace repository
                print("Loading LongBench v2 dataset from HuggingFace: zai-org/LongBench-v2")
                dataset = load_dataset('zai-org/LongBench-v2', split='train')
                data_all = list(dataset)
                
        except Exception as e:
            print(f"Failed to load LongBench v2 dataset: {e}")
            return
        
        # Filter by difficulty if specified
        if hasattr(self.eval_args, 'longbench_difficulty') and self.eval_args.longbench_difficulty:
            difficulty = self.eval_args.longbench_difficulty
            data_all = [item for item in data_all if item.get('difficulty') == difficulty]
            print(f"Filtered to difficulty: {difficulty}, {len(data_all)} examples")
        
        # Limit examples if specified
        if hasattr(self.eval_args, 'longbench_max_examples') and self.eval_args.longbench_max_examples:
            max_examples = self.eval_args.longbench_max_examples
            data_all = data_all[:max_examples]
            print(f"Limited to {len(data_all)} examples")
        
        # Prepare output
        output_dir = self.eval_args.save_dir or "results/longbench_v2"
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        correct = 0
        total = 0
        
        # Evaluate each example
        for item in tqdm(data_all, desc="Evaluating"):
            # Format prompt
            prompt = self.format_prompt(item)
            
            # Generate answer
            response = self.generate_answer(prompt)
            
            # Extract predicted answer
            pred_answer = self.extract_answer(response)
            
            # Get ground truth
            true_answer = item['answer']
            
            # Check if correct
            is_correct = pred_answer == true_answer
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            result = {
                "_id": item.get("_id", f"item_{total}"),
                "domain": item.get("domain", ""),
                "sub_domain": item.get("sub_domain", ""),
                "difficulty": item.get("difficulty", ""),
                "question": item["question"],
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "response": response[:500],  # Truncate for storage
                "correct": is_correct
            }
            results.append(result)
            
            # Save intermediate results
            if total % 10 == 0:
                with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Progress: {correct}/{total} = {correct/total*100:.2f}%")
        
        # Final results
        accuracy = correct / total if total > 0 else 0
        
        # Save final results
        final_results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
        
        with open(os.path.join(output_dir, "final_results.json"), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("LongBench v2 Evaluation Results:")
        print(f"Total Examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Results saved to: {output_dir}")
        print("="*60)


def run_longbench_v2_direct_eval() -> None:
    """Entry point for direct LongBench v2 evaluation."""
    evaluator = LongBenchV2DirectEvaluator()
    evaluator.eval()