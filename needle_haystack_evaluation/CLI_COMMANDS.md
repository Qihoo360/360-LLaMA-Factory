# CLI Commands for Needle in Haystack Evaluation

This document provides various command-line options for running needle in haystack evaluation using LlamaFactory.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Navigate to the 360-LLaMA-Factory root directory before running commands

## Command Line Options

### Option 1: After Installing LlamaFactory

If you've installed LlamaFactory as a package:

```bash
# Install LlamaFactory first
cd /path/to/360-LLaMA-Factory
pip install -e .

# Run evaluation with different configurations
llamafactory-cli eval needle_haystack_evaluation/needle_haystack_config.yaml
llamafactory-cli eval paulgraham_demo_config.yaml
llamafactory-cli eval quick_niah_demo.yaml
llamafactory-cli eval custom_directory_demo_config.yaml
```

### Option 2: Direct Python Execution (No Installation Required)

Run evaluations directly without installing LlamaFactory:

#### Standard Evaluation
```bash
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'needle_haystack_evaluation/needle_haystack_config.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

#### Paul Graham Essays
```bash
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'paulgraham_demo_config.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

#### Quick Demo
```bash
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'quick_niah_demo.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

#### Custom Directory
```bash
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'custom_directory_demo_config.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

### Option 3: One-Line Commands

Compact versions for quick execution:

```bash
# Standard tech background
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','needle_haystack_evaluation/needle_haystack_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"

# Paul Graham essays
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','paulgraham_demo_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"

# Quick demo
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','quick_niah_demo.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"
```

### Option 4: Shell Script Execution

Use the provided shell scripts:

```bash
# Standard evaluation script
bash needle_haystack_evaluation/run_evaluation.sh

# Quick demo script (includes installation)
chmod +x quick_niah_demo.sh
./quick_niah_demo.sh
```

### Option 5: Python Script Execution

Use the dedicated Python runner:

```bash
# Run with Python script
python3 run_needle_eval.py
```

## Configuration Files

### Available Configurations

1. **Standard Configuration** (`needle_haystack_evaluation/needle_haystack_config.yaml`)
   - Uses custom technology-focused background text
   - Context lengths: [240, 480, 958, 1918] tokens
   - Needle positions: [0%, 25%, 50%, 75%, 100%]

2. **Paul Graham Essays** (`paulgraham_demo_config.yaml`)
   - Uses real Paul Graham essays for background
   - More realistic and diverse text
   - Context lengths: [500, 1000] tokens

3. **Quick Demo** (`quick_niah_demo.yaml`)
   - Minimal configuration for quick testing
   - Context lengths: [100, 200] tokens
   - Only 3 needle positions: [0%, 50%, 100%]

4. **Custom Directory** (`custom_directory_demo_config.yaml`)
   - Load background text from any directory
   - Specify path with `needle_haystack_data_dir`

### Creating Custom Configuration

Create a new YAML file with these parameters:

```yaml
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
finetuning_type: full
task: needle_haystack_proper
task_dir: evaluation
template: alpaca
save_dir: saves/my_custom_eval
batch_size: 1

# Needle haystack specific
needle_context_lengths: [500, 1000, 2000]
needle_depth_percents: [0, 25, 50, 75, 100]
needle_text: "The answer is XYZ789."
needle_question: "What is the answer?"
needle_haystack_data_source: "custom"  # or "paulgraham" or "directory"
# needle_haystack_data_dir: "/path/to/text/files"  # for directory mode
```

Run with:
```bash
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','my_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"
```

## Command Options and Parameters

### Environment Variables

```bash
# Run on CPU only
export CUDA_VISIBLE_DEVICES=""

# Specify GPU
export CUDA_VISIBLE_DEVICES="0"

# Set cache directory
export HF_HOME="/path/to/cache"
```

### Running Multiple Evaluations

```bash
# Batch evaluation script
for config in *_config.yaml; do
    echo "Running $config"
    python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','$config'];from llamafactory.eval.evaluator import run_eval;run_eval()"
done
```

## Troubleshooting

### Common Issues

1. **Module not found error**
   ```bash
   # Make sure you're in the repository root
   cd /path/to/360-LLaMA-Factory
   ```

2. **CUDA out of memory**
   ```bash
   # Reduce batch size in config
   batch_size: 1
   
   # Or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Permission denied**
   ```bash
   # Make scripts executable
   chmod +x quick_niah_demo.sh
   chmod +x needle_haystack_evaluation/run_evaluation.sh
   ```

4. **Model download issues**
   ```bash
   # Set proxy if needed
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

## Expected Output

Successful execution shows:

```
[INFO] Loading model...
[INFO] Running needle haystack evaluation...
Processing examples: 100%|██████████| 20/20 [00:45<00:00]

=== Needle in Haystack Evaluation Results ===
Overall Average Score: 0.988
Exact Match Rate: 0.950
Total Examples: 20

Scores by Context Length:
  240 tokens: 1.000
  480 tokens: 0.950
  958 tokens: 1.000
  1,918 tokens: 1.000

Scores by Needle Depth:
  0%: 1.000
  25%: 0.938
  50%: 1.000
  75%: 1.000
  100%: 1.000
```

Results saved in the configured `save_dir`:
- `summary.json` - Overall metrics
- `detailed_results.json` - Individual results
- `needle_haystack_performance.png` - Charts
- `needle_haystack_heatmap.png` - Heatmap

## Analysis and Comparison

After running evaluations, use the comparison plotting script to analyze results:

### Generate Comprehensive Comparison Report
```bash
python3 plot_niah_comparison.py --plot-type all --output-dir comparison_plots
```

### Compare Specific Experiments
```bash
python3 plot_niah_comparison.py --experiments experiment1 experiment2 --plot-type all
```

### Generate Individual Plot Types
```bash
# Overall performance comparison
python3 plot_niah_comparison.py --plot-type overall

# Context length analysis
python3 plot_niah_comparison.py --plot-type context

# Needle depth analysis
python3 plot_niah_comparison.py --plot-type depth

# Specific experiment heatmap
python3 plot_niah_comparison.py --plot-type heatmap --experiment-for-heatmap experiment_name
```

See `evaluation/needle_haystack/COMPARISON_PLOTTING.md` for detailed documentation.