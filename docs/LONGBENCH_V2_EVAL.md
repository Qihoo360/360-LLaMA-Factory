# LongBench v2 Evaluation Guide

## Overview

The unified LongBench v2 evaluation system supports multiple backends and automatic template detection for maximum flexibility and compatibility.

## Key Features

### 1. Multiple Evaluation Backends

- **Direct Mode** (default): Loads model directly, uses native chat templates
- **vLLM Mode**: Uses vLLM server for high-performance inference  
- **Official Mode**: Uses official LongBench scripts with vLLM server

### 2. Automatic Template Detection

Automatically detects and uses the model's native `chat_template` from `tokenizer_config.json`:
- **Perfect compatibility** with the model's expected format
- **No manual configuration** needed for most models
- **Fallbacks** to manual templates when needed

### 3. How Backend Selection Works

The evaluator automatically chooses the best backend:

1. **Check for explicit mode**: If `longbench_mode` specified in config
2. **Check for vLLM server**: If server accessible, uses vLLM mode
3. **Default to direct**: Loads model directly (most common)

## Configuration

### Basic Configuration (Auto-Detection)
```yaml
# eval_configs/longbench_v2_basic.yaml
model_name_or_path: your-model-path
# template: llama3  # Optional - auto-detects from tokenizer_config.json

task: longbench_v2
save_dir: results/longbench_v2
```

### Force Specific Backend

**Option 1: Environment Variable (works with any repo version)**
```bash
export LONGBENCH_MODE=direct  # Options: direct, vllm, official
llamafactory-cli eval eval_configs/longbench_v2_basic.yaml
```

**Option 2: YAML Parameter (requires updated evaluation_args.py)**
```yaml
model_name_or_path: your-model-path
longbench_mode: vllm    # Options: direct, vllm, official
```

**Option 3: Save Directory Hint (for backward compatibility)**
```yaml
save_dir: results/longbench_v2_direct  # Contains 'direct' -> uses direct mode
save_dir: results/longbench_v2_vllm    # Default -> uses vLLM mode
save_dir: results/longbench_v2_official # Contains 'official' -> uses official mode
```

### vLLM Server Configuration
```yaml
# Environment variables (optional)
# VLLM_URL: http://127.0.0.1:8000/v1
# VLLM_API_KEY: token-abc123

longbench_mode: vllm
```

### Official Scripts Mode
```yaml
longbench_mode: official  # Uses third_party/LongBench/pred.py
```

## Usage Examples

### 1. Evaluate with Auto-Detection
```bash
llamafactory-cli eval eval_configs/longbench_v2_basic.yaml
```

### 2. Evaluate with Local Dataset
```yaml
task: longbench_v2
task_dir: /path/to/local/longbench-v2-dataset
```

### 3. Limit Examples for Testing
```yaml
longbench_max_examples: 10  # Only evaluate first 10 examples
```

## Supported Models

Works automatically with any model that has a `chat_template` in its tokenizer config, including:
- Llama 2/3 models
- Mistral/Mixtral models  
- Qwen models
- DeepSeek models
- Gemma models
- Any HuggingFace model with proper chat_template

## Dataset Sources

The evaluator supports multiple dataset sources:

1. **HuggingFace Hub** (default): `zai-org/LongBench-v2`
2. **Local Directory**: Specify via `task_dir` parameter
3. **Custom HuggingFace Dataset**: Specify dataset ID in `task_dir`

## Output

Results are saved to the specified `save_dir` with:
- `results.json`: Detailed results for each example
- Accuracy metrics
- Per-example predictions and correctness

## Advanced Options

### Generation Parameters
```yaml
longbench_temperature: 0.1
longbench_top_p: 1.0
longbench_top_k: 1
longbench_max_new_tokens: 128
```

### Filtering Options
```yaml
longbench_difficulty: easy  # or "hard"
longbench_domains: ["single_doc", "multi_doc"]  # specific domains
```

## Troubleshooting

### No Chat Template Found
If your model doesn't have a chat_template, either:
1. Specify a template manually in the YAML config
2. Add a chat_template to the model's tokenizer_config.json

### Wrong Format Used
If auto-detection uses the wrong format:
1. Override by specifying `template: your_template` in the config
2. Check if the model's tokenizer_config.json has the correct chat_template

### Memory Issues
For large models or long contexts:
```yaml
low_cpu_mem_usage: true
infer_dtype: bfloat16  # or float16
longbench_max_context_length: 16000  # reduce if needed
```