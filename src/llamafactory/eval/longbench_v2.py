"""LongBench v2 evaluation with multiple backend support."""

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pathlib import Path
from enum import Enum

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class EvalMode(Enum):
    """Evaluation backend modes."""
    DIRECT = "direct"  # Load model directly
    VLLM = "vllm"  # Use vLLM server
    OFFICIAL = "official"  # Use official scripts


class LongBenchV2Evaluator:
    """Unified LongBench v2 evaluator with multiple backend support."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize evaluator."""
        self.model_args, self.data_args, self.eval_args, self.finetuning_args = get_eval_args(args)
        
        # Determine evaluation mode
        self.mode = self._determine_mode()
        print(f"Using evaluation mode: {self.mode.value}")
        
        # Initialize based on mode
        if self.mode == EvalMode.DIRECT:
            self._init_direct()
        elif self.mode == EvalMode.VLLM:
            self._init_vllm()
        elif self.mode == EvalMode.OFFICIAL:
            self._init_official()
    
    def _determine_mode(self) -> EvalMode:
        """Determine which evaluation mode to use."""
        # Check if user specified a mode
        if hasattr(self.eval_args, 'longbench_mode'):
            mode = self.eval_args.longbench_mode.lower()
            if mode in ["vllm", "server"]:
                return EvalMode.VLLM
            elif mode == "official":
                return EvalMode.OFFICIAL
        
        # Check if vLLM server is running
        if self._check_vllm_server():
            return EvalMode.VLLM
        
        # Default to direct mode
        return EvalMode.DIRECT
    
    def _check_vllm_server(self) -> bool:
        """Check if vLLM server is accessible."""
        try:
            import requests
            url = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/models")
            response = requests.get(url, timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def _init_direct(self) -> None:
        """Initialize for direct model loading."""
        # Load tokenizer
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"
        
        # Check for native chat template
        if self.tokenizer.chat_template is not None:
            print(f"Using model's native chat_template from tokenizer_config.json")
            self.use_chat_template = True
        else:
            print(f"No chat_template found, using template: {self.data_args.template or 'default'}")
            self.use_chat_template = False
            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        
        # Load model
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args)
        self.model.eval()
    
    def _init_vllm(self) -> None:
        """Initialize for vLLM server usage."""
        from openai import OpenAI
        
        # Setup client
        self.vllm_url = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "token-abc123")
        self.client = OpenAI(base_url=self.vllm_url, api_key=self.vllm_api_key)
        
        # Get model name
        self.model_name = os.path.basename(self.model_args.model_name_or_path)
        
        # Load tokenizer for truncation
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        
        print(f"Connected to vLLM server at {self.vllm_url}")
    
    def _init_official(self) -> None:
        """Initialize for official script usage."""
        self.longbench_dir = Path("third_party/LongBench")
        if not self.longbench_dir.exists():
            raise RuntimeError(
                f"LongBench repository not found at {self.longbench_dir}. "
                "Please clone: git clone https://github.com/THUDM/LongBench.git third_party/LongBench"
            )
        
        self.pred_script = self.longbench_dir / "pred.py"
        self.result_script = self.longbench_dir / "result.py"
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load LongBench v2 dataset."""
        try:
            # Check for local dataset
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
                    print(f"Loading from HuggingFace: {dataset_path}")
                    dataset = load_dataset(dataset_path, split='train')
                    data_all = list(dataset)
            else:
                # Default to HuggingFace
                print("Loading from HuggingFace: zai-org/LongBench-v2")
                dataset = load_dataset('zai-org/LongBench-v2', split='train')
                data_all = list(dataset)
            
            # Apply filters
            if hasattr(self.eval_args, 'longbench_difficulty') and self.eval_args.longbench_difficulty:
                difficulty = self.eval_args.longbench_difficulty
                data_all = [item for item in data_all if item.get('difficulty') == difficulty]
                print(f"Filtered to difficulty: {difficulty}, {len(data_all)} examples")
            
            if hasattr(self.eval_args, 'longbench_max_examples') and self.eval_args.longbench_max_examples:
                data_all = data_all[:self.eval_args.longbench_max_examples]
                print(f"Limited to {len(data_all)} examples")
            
            return data_all
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format prompt based on mode and template."""
        question_text = f"""{item['question']}

Choose the correct answer from the options below:
(A) {item['choice_A']}
(B) {item['choice_B']}  
(C) {item['choice_C']}
(D) {item['choice_D']}

The correct answer is ("""
        
        if self.mode == EvalMode.DIRECT and self.use_chat_template:
            # Use native chat template
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
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Ensure prompt ends correctly
            if not prompt.rstrip().endswith("The correct answer is ("):
                if prompt.rstrip().endswith(":") or prompt.rstrip().endswith("\n"):
                    prompt = prompt.rstrip() + " The correct answer is ("
                else:
                    prompt = prompt + "The correct answer is ("
        else:
            # Use manual template or default format
            context = self.truncate_context(item['context'], 32000)
            
            # Load official template if available
            template_path = Path("third_party/LongBench/prompts/0shot.txt")
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    template = f.read()
                    prompt = template.replace('$DOC$', context)
                    prompt = prompt.replace('$Q$', item['question'])
                    prompt = prompt.replace('$C_A$', item['choice_A'])
                    prompt = prompt.replace('$C_B$', item['choice_B'])
                    prompt = prompt.replace('$C_C$', item['choice_C'])
                    prompt = prompt.replace('$C_D$', item['choice_D'])
            else:
                prompt = f"""Given the document below, answer the following question.

Document:
{context}

{question_text}"""
        
        return prompt
    
    def truncate_context(self, text: str, max_length: int) -> str:
        """Truncate context using middle truncation."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return text
        
        half_length = max_length // 2
        truncated_tokens = tokens[:half_length] + tokens[-half_length:]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from response."""
        response = response.strip()
        
        # Try patterns
        patterns = [
            r'^([A-D])',
            r'answer is \(?([A-D])\)?',
            r'correct answer is \(?([A-D])\)?',
            r'\(?([A-D])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Check first character
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
        
        return None
    
    @torch.no_grad()
    def generate_direct(self, prompt: str) -> str:
        """Generate using direct model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128000)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": 5,
            "temperature": 0.1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    def generate_vllm(self, prompt: str) -> str:
        """Generate using vLLM server."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5,
        )
        return completion.choices[0].message.content
    
    def eval(self) -> None:
        """Run evaluation."""
        print(f"Starting LongBench v2 evaluation in {self.mode.value} mode...")
        
        # Load dataset
        data_all = self.load_dataset()
        
        # Prepare output
        output_dir = self.eval_args.save_dir or "results/longbench_v2"
        os.makedirs(output_dir, exist_ok=True)
        
        # For official mode, delegate to official scripts
        if self.mode == EvalMode.OFFICIAL:
            self._run_official_eval(data_all, output_dir)
            return
        
        # Direct or vLLM evaluation
        results = []
        correct = 0
        total = 0
        
        for item in tqdm(data_all, desc="Evaluating"):
            # Format prompt
            prompt = self.format_prompt(item)
            
            # Generate response
            if self.mode == EvalMode.DIRECT:
                response = self.generate_direct(prompt)
            elif self.mode == EvalMode.VLLM:
                response = self.generate_vllm(prompt)
            
            # Extract and check answer
            pred_answer = self.extract_answer(response)
            true_answer = item['answer']
            is_correct = pred_answer == true_answer
            
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                "_id": item.get("_id", f"item_{total}"),
                "domain": item.get("domain", ""),
                "sub_domain": item.get("sub_domain", ""),
                "difficulty": item.get("difficulty", ""),
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "correct": is_correct
            })
            
            # Save intermediate results
            if total % 10 == 0:
                self._save_results(results, correct, total, output_dir)
                print(f"Progress: {correct}/{total} = {correct/total*100:.2f}%")
        
        # Save final results
        self._save_results(results, correct, total, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("LongBench v2 Evaluation Results:")
        print(f"Mode: {self.mode.value}")
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {correct/total*100:.2f}%")
        print(f"Results saved to: {output_dir}")
        print("="*60)
    
    def _save_results(self, results: List[Dict], correct: int, total: int, output_dir: str) -> None:
        """Save evaluation results."""
        final_results = {
            "mode": self.mode.value,
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "results": results
        }
        
        with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    def _run_official_eval(self, data_all: List[Dict], output_dir: str) -> None:
        """Run evaluation using official scripts."""
        # Save dataset for official scripts
        data_file = os.path.join(output_dir, "data.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_all, f, ensure_ascii=False, indent=2)
        
        # Update model config
        config_dir = self.longbench_dir / "config"
        model2path_file = config_dir / "model2path.json"
        
        if model2path_file.exists():
            with open(model2path_file, 'r') as f:
                model_paths = json.load(f)
        else:
            model_paths = {}
        
        model_name = os.path.basename(self.model_args.model_name_or_path)
        model_paths[model_name] = self.model_args.model_name_or_path
        
        with open(model2path_file, 'w') as f:
            json.dump(model_paths, f, indent=2)
        
        # Run official prediction
        output_file = os.path.join(output_dir, "predictions.json")
        cmd = [
            sys.executable,
            str(self.pred_script),
            "--model", model_name,
            "--data", data_file,
            "--output", output_file
        ]
        
        print(f"Running official evaluation: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Process results
            if self.result_script.exists() and os.path.exists(output_file):
                result_cmd = [
                    sys.executable,
                    str(self.result_script),
                    "--predictions", output_file,
                    "--output", os.path.join(output_dir, "final_results.json")
                ]
                
                result = subprocess.run(result_cmd, capture_output=True, text=True, check=True)
                print("\nResults:")
                print(result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running official evaluation: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print("\nTo run manually:")
            print(f"1. Start vLLM: vllm serve {self.model_args.model_name_or_path}")
            print(f"2. Run: {' '.join(cmd)}")


def run_longbench_eval() -> None:
    """Entry point for LongBench v2 evaluation."""
    evaluator = LongBenchV2Evaluator()
    evaluator.eval()