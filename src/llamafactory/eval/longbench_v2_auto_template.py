"""LongBench v2 evaluation with automatic template detection."""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class LongBenchV2AutoTemplateEvaluator:
    """LongBench v2 evaluator that auto-detects and uses model's chat template."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize evaluator with model and tokenizer."""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # For batch generation
        
        # Check if tokenizer has a chat template
        if self.tokenizer.chat_template is not None:
            print(f"Using model's native chat_template from tokenizer_config.json")
            self.use_chat_template = True
        else:
            print(f"No chat_template found in tokenizer, using manual formatting")
            self.use_chat_template = False
            # Load the template if specified, otherwise use default
            from ..data import get_template_and_fix_tokenizer
            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        
        # Load model
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.model.eval()
        
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format the prompt using either chat template or manual formatting."""
        
        # Create the question with choices
        question_text = f"""{item['question']}

Choose the correct answer from the options below:
(A) {item['choice_A']}
(B) {item['choice_B']}
(C) {item['choice_C']}
(D) {item['choice_D']}

The correct answer is ("""
        
        if self.use_chat_template:
            # Use the tokenizer's apply_chat_template
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the multiple choice question based on the given document."
                },
                {
                    "role": "user", 
                    "content": f"Document:\n{item['context']}\n\n{question_text}"
                }
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Some templates already add the start of assistant response
            # We want to end with "The correct answer is ("
            if not prompt.rstrip().endswith("The correct answer is ("):
                # Add our prompt ending if needed
                if prompt.rstrip().endswith(":") or prompt.rstrip().endswith("\n"):
                    prompt = prompt.rstrip() + " The correct answer is ("
                else:
                    prompt = prompt + "The correct answer is ("
        else:
            # Manual formatting (fallback)
            context = self.truncate_context(item['context'], 32000)
            prompt = f"""Given the document below, answer the following question.

Document:
{context}

{question_text}"""
        
        return prompt
    
    def truncate_context(self, text: str, max_length: int) -> str:
        """Truncate context to fit within max length."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return text
        
        # Middle truncation
        half_length = max_length // 2
        truncated_tokens = tokens[:half_length] + tokens[-half_length:]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from model response."""
        response = response.strip()
        
        # Check if response starts with A, B, C, or D
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
        
        return None
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer from the model."""
        import torch
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128000)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": 5,  # We only need the letter
            "temperature": 0.1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def eval(self) -> None:
        """Run evaluation."""
        print("Starting LongBench v2 evaluation with auto-template detection...")
        
        # Load dataset
        try:
            if hasattr(self.eval_args, 'task_dir') and self.eval_args.task_dir:
                dataset_path = self.eval_args.task_dir
                if os.path.exists(dataset_path):
                    print(f"Loading from local path: {dataset_path}")
                    if dataset_path.endswith('.json'):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            data_all = json.load(f)
                    else:
                        dataset = load_dataset(dataset_path, split='train')
                        data_all = list(dataset)
                else:
                    dataset = load_dataset(dataset_path, split='train')
                    data_all = list(dataset)
            else:
                print("Loading from HuggingFace: zai-org/LongBench-v2")
                dataset = load_dataset('zai-org/LongBench-v2', split='train')
                data_all = list(dataset)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return
        
        # Limit examples if specified
        if hasattr(self.eval_args, 'longbench_max_examples'):
            data_all = data_all[:self.eval_args.longbench_max_examples]
        
        # Evaluate
        results = []
        correct = 0
        total = 0
        
        for item in tqdm(data_all, desc="Evaluating"):
            prompt = self.format_prompt(item)
            response = self.generate_answer(prompt)
            pred_answer = self.extract_answer(response)
            true_answer = item['answer']
            
            is_correct = pred_answer == true_answer
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "_id": item.get("_id", f"item_{total}"),
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "correct": is_correct
            })
        
        # Save results
        output_dir = self.eval_args.save_dir or "results/longbench_v2"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump({
                "accuracy": correct / total if total > 0 else 0,
                "correct": correct,
                "total": total,
                "results": results
            }, f, indent=2)
        
        print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.2f}%")


def run_longbench_v2_auto_eval() -> None:
    """Entry point for auto-template evaluation."""
    evaluator = LongBenchV2AutoTemplateEvaluator()
    evaluator.eval()