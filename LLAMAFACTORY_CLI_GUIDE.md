# LlamaFactory CLI Guide for Needle Haystack Evaluation

## Installation

First, install LlamaFactory in your environment:

```bash
cd /path/to/360-LLaMA-Factory
pip install -e .
```

## Running Needle Haystack Evaluation with CLI

### Method 1: Using llamafactory-cli (after installation)

```bash
# Basic evaluation with default settings
llamafactory-cli eval needle_haystack_evaluation/needle_haystack_config.yaml

# With Paul Graham essays
llamafactory-cli eval paulgraham_demo_config.yaml

# Quick demo
llamafactory-cli eval quick_niah_demo.yaml

# Custom directory
llamafactory-cli eval custom_directory_demo_config.yaml
```

### Method 2: Direct Python execution (no installation needed)

```bash
# From the repository root
cd /path/to/360-LLaMA-Factory

# Basic evaluation
python -m llamafactory.cli eval needle_haystack_evaluation/needle_haystack_config.yaml

# With Paul Graham essays
python -m llamafactory.cli eval paulgraham_demo_config.yaml

# Quick demo
python -m llamafactory.cli eval quick_niah_demo.yaml
```

### Method 3: Using src/llamafactory directly

```bash
cd /path/to/360-LLaMA-Factory
python src/llamafactory/cli.py eval needle_haystack_evaluation/needle_haystack_config.yaml
```

## Command Line Options

The eval command accepts a YAML configuration file with all parameters:

```bash
llamafactory-cli eval [CONFIG_FILE] [OPTIONS]
```

### Available Options:
- `--help` or `-h`: Show help message
- Configuration file (required): Path to YAML config

## Example Commands with Different Configurations

### 1. Standard Evaluation
```bash
llamafactory-cli eval needle_haystack_evaluation/needle_haystack_config.yaml
```

### 2. Paul Graham Essays (Realistic Text)
```bash
llamafactory-cli eval paulgraham_demo_config.yaml
```

### 3. Quick Demo (Minimal Test)
```bash
llamafactory-cli eval quick_niah_demo.yaml
```

### 4. Custom Text Directory
```bash
# First, create a config file pointing to your text directory
cat > my_custom_config.yaml << EOF
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
finetuning_type: full
task: needle_haystack_proper
task_dir: evaluation
template: alpaca
save_dir: saves/my_custom_eval
batch_size: 1
needle_context_lengths: [500, 1000]
needle_depth_percents: [0, 50, 100]
needle_text: "The code is ABC123."
needle_question: "What is the code?"
needle_haystack_data_source: "directory"
needle_haystack_data_dir: "/path/to/my/text/files"
EOF

# Run evaluation
llamafactory-cli eval my_custom_config.yaml
```

## Configuration Parameters

All parameters are set in the YAML configuration file:

```yaml
### Model Configuration
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0

### Method
finetuning_type: full

### Dataset
task: needle_haystack_proper
task_dir: evaluation
template: alpaca
lang: en

### Output
save_dir: saves/tinyllama/needle_haystack_eval

### Evaluation
batch_size: 1

### Needle Haystack Specific
needle_context_lengths: [240, 480, 958, 1918]
needle_depth_percents: [0, 25, 50, 75, 100]
needle_text: "The secret key is 42 alpha bravo."
needle_question: "What is the secret key?"
needle_haystack_data_source: "custom"  # Options: "custom", "paulgraham", "directory"
# needle_haystack_data_dir: "/path/to/text/files"  # For directory mode
```

## Expected Output

When running the evaluation, you'll see:

```
[INFO] Loading model TinyLlama/TinyLlama-1.1B-Chat-v1.0...
[INFO] Running needle haystack evaluation...
Processing examples: 100%|██████████| 20/20 [00:45<00:00,  2.25s/it]

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

## Results Location

Results are saved in the directory specified by `save_dir`:
- `summary.json` - Overall metrics
- `detailed_results.json` - Individual example results
- `needle_haystack_performance.png` - Performance charts
- `needle_haystack_heatmap.png` - Heatmap visualization

## Troubleshooting

### Command not found
```bash
# Install LlamaFactory first
pip install -e .

# Or use direct Python execution
python -m llamafactory.cli eval config.yaml
```

### Module not found
```bash
# Make sure you're in the repository root
cd /path/to/360-LLaMA-Factory

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### CUDA/GPU issues
```bash
# Run on CPU only
export CUDA_VISIBLE_DEVICES=""
llamafactory-cli eval config.yaml
```

### Memory issues
```bash
# Reduce batch size in config
batch_size: 1

# Use smaller model
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Advanced Usage

### Running Multiple Evaluations
```bash
# Create a script to run multiple configs
for config in *_config.yaml; do
    echo "Running $config"
    llamafactory-cli eval "$config"
done
```

### Custom Model Evaluation
```yaml
# Use any HuggingFace model
model_name_or_path: meta-llama/Llama-2-7b-chat-hf
# Or local model
model_name_or_path: /path/to/local/model
```

### Different Templates
```yaml
# Available templates
template: alpaca  # Default
template: llama3
template: vicuna
template: chatml
```