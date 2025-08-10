# LongBench v2 Evaluation Guide

## Overview

The LongBench v2 evaluation system now supports automatic chat template detection, making it easier to evaluate models without needing to manually specify the correct template format.

## Key Features

### 1. Automatic Template Detection

The evaluator automatically detects and uses the model's native `chat_template` from `tokenizer_config.json`. This ensures:
- **Perfect compatibility** with the model's expected format
- **No manual configuration** needed for most models
- **Automatic handling** of custom chat formats

### 2. How It Works

When you run an evaluation:

1. **First Priority**: Uses the model's `chat_template` if present in `tokenizer_config.json`
2. **Fallback**: If no chat template exists, uses the template specified in your YAML config
3. **Default**: If neither exists, uses a generic format

### 3. Configuration

#### Basic Configuration (Auto-Detection)
```yaml
# eval_configs/longbench_v2_basic.yaml
model_name_or_path: your-model-path
# template: llama3  # Optional - omit for auto-detection

task: longbench_v2
save_dir: results/longbench_v2
```

#### Force Specific Template
```yaml
model_name_or_path: your-model-path
template: llama3  # Forces use of llama3 template, ignoring tokenizer_config.json
```

#### Use Empty Template (Preserves Original)
```yaml
model_name_or_path: your-model-path
template: empty  # Uses tokenizer's chat_template without modification
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