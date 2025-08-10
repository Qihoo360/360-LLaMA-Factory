"""LongBench v2 evaluation wrapper using official evaluation scripts."""

import json
import os
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Optional
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from ..data import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    pass


class LongBenchV2Wrapper:
    """Wrapper for running official LongBench v2 evaluation."""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize wrapper with model configuration."""
        self.model_args, self.data_args, self.eval_args, self.finetuning_args = get_eval_args(args)
        
        # Path to official evaluation scripts
        self.longbench_dir = Path("third_party/LongBench")
        if not self.longbench_dir.exists():
            raise RuntimeError(
                f"LongBench repository not found at {self.longbench_dir}. "
                "Please clone it: git clone https://github.com/THUDM/LongBench.git third_party/LongBench"
            )
        
        # Check for required files
        self.pred_script = self.longbench_dir / "pred.py"
        self.result_script = self.longbench_dir / "result.py"
        
        if not self.pred_script.exists():
            raise RuntimeError(f"Official evaluation script not found: {self.pred_script}")
            
    def prepare_model_server(self) -> Dict[str, Any]:
        """Prepare model for serving via API (for compatibility with official scripts)."""
        # Load model and tokenizer
        tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        tokenizer.padding_side = "right"
        
        template = get_template_and_fix_tokenizer(tokenizer, self.data_args)
        model = load_model(tokenizer, self.model_args, self.finetuning_args)
        
        # Return model info for potential vLLM or other server setup
        return {
            "model": model,
            "tokenizer": tokenizer,
            "template": template,
            "model_path": self.model_args.model_name_or_path
        }
    
    def download_dataset(self) -> str:
        """Download or locate LongBench v2 dataset."""
        # Check if task_dir is specified for local dataset
        if hasattr(self.eval_args, 'task_dir') and self.eval_args.task_dir:
            dataset_path = self.eval_args.task_dir
            if os.path.exists(dataset_path):
                print(f"Using local dataset from: {dataset_path}")
                return dataset_path
        
        # Download from HuggingFace
        print("Downloading LongBench v2 dataset from HuggingFace...")
        dataset = load_dataset('zai-org/LongBench-v2', split='train')
        
        # Save to temporary directory for official scripts
        temp_dir = tempfile.mkdtemp(prefix="longbench_v2_")
        data_file = os.path.join(temp_dir, "data.json")
        
        # Format data for official evaluation
        data_formatted = []
        for item in dataset:
            data_formatted.append({
                "_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "question": item["question"],
                "choice_A": item["choice_A"],
                "choice_B": item["choice_B"],
                "choice_C": item["choice_C"],
                "choice_D": item["choice_D"],
                "answer": item["answer"],
                "context": item["context"]
            })
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_formatted, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to: {data_file}")
        return data_file
    
    def run_evaluation(self) -> None:
        """Run the official LongBench v2 evaluation."""
        print("Starting LongBench v2 evaluation using official scripts...")
        
        # 1. Download/locate dataset
        dataset_path = self.download_dataset()
        
        # 2. Prepare model info
        model_info = self.prepare_model_server()
        
        # 3. Create configuration for official scripts
        config_dir = self.longbench_dir / "config"
        
        # Update model path configuration
        model2path_file = config_dir / "model2path.json"
        if model2path_file.exists():
            with open(model2path_file, 'r') as f:
                model_paths = json.load(f)
        else:
            model_paths = {}
        
        # Add our model to the config
        model_name = os.path.basename(self.model_args.model_name_or_path)
        model_paths[model_name] = self.model_args.model_name_or_path
        
        # Save updated config
        with open(model2path_file, 'w') as f:
            json.dump(model_paths, f, indent=2)
        
        # 4. Prepare output directory
        output_dir = self.eval_args.save_dir or "results/longbench_v2"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "predictions.json")
        
        # 5. Run official prediction script
        print(f"Running predictions with model: {model_name}")
        
        # Build command for official script
        cmd = [
            sys.executable,
            str(self.pred_script),
            "--model", model_name,
            "--data", dataset_path,
            "--output", output_file
        ]
        
        # Add optional parameters
        if hasattr(self.eval_args, 'longbench_max_examples'):
            cmd.extend(["--max_examples", str(self.eval_args.longbench_max_examples)])
        
        if hasattr(self.eval_args, 'longbench_temperature'):
            cmd.extend(["--temperature", str(self.eval_args.longbench_temperature)])
            
        # Run the evaluation
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Note: In practice, you might need to start a vLLM server first
            # For now, this shows the structure
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # 6. Compute results using official script
            if self.result_script.exists() and os.path.exists(output_file):
                result_cmd = [
                    sys.executable,
                    str(self.result_script),
                    "--predictions", output_file,
                    "--output", os.path.join(output_dir, "results.json")
                ]
                
                result = subprocess.run(result_cmd, capture_output=True, text=True, check=True)
                print("\nEvaluation Results:")
                print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running official evaluation: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            
            # Fallback: Provide instructions for manual evaluation
            print("\n" + "="*60)
            print("To run evaluation manually:")
            print(f"1. Start vLLM server with your model: {self.model_args.model_name_or_path}")
            print(f"2. Run: python {self.pred_script} --model {model_name} --data {dataset_path}")
            print(f"3. Run: python {self.result_script} --predictions {output_file}")
            print("="*60)


def run_longbench_v2_eval() -> None:
    """Entry point for LongBench v2 evaluation."""
    evaluator = LongBenchV2Wrapper()
    evaluator.run_evaluation()