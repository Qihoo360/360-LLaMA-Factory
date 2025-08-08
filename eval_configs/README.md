# Needle Haystack Evaluation Configurations

Production YAML configurations for needle-in-haystack evaluation with RoPE extensions and multi-GPU support.

## Available Configurations

- **needle_haystack_h100_8gpu.yaml**: YARN RoPE with 8-GPU sequence parallelism
- **needle_haystack_longrope.yaml**: LongRoPE scaling for extreme context lengths  
- **needle_haystack_llama3_rope.yaml**: Llama3 RoPE scaling
- **needle_haystack_advanced_generation.yaml**: Advanced generation parameters
- **needle_haystack_test.yaml**: Quick test configuration

## Usage

```bash
llamafactory-cli eval eval_configs/needle_haystack_h100_8gpu.yaml
llamafactory-cli eval eval_configs/needle_haystack_longrope.yaml
llamafactory-cli eval eval_configs/needle_haystack_llama3_rope.yaml
```

## Key Features

- RoPE scaling: linear, dynamic, yarn, longrope, llama3
- Multi-GPU sequence parallelism (1-8 GPUs)
- Generation config integration
- Chat template support
- Input/output saving
- Configurable context lengths and needle positions