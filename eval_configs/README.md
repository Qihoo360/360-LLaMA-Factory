# Needle Haystack Evaluation with RoPE Extensions

This directory contains YAML configuration files for running needle-in-haystack evaluations using `llamafactory-cli eval` with various RoPE scaling techniques.

## Usage

```bash
# Run needle haystack evaluation with YARN RoPE on 8 H100s
llamafactory-cli eval eval_configs/needle_haystack_h100_8gpu.yaml

# Run with LongRoPE scaling
llamafactory-cli eval eval_configs/needle_haystack_longrope.yaml  

# Run with Llama3 RoPE scaling
llamafactory-cli eval eval_configs/needle_haystack_llama3_rope.yaml
```

## Configuration Files

- **needle_haystack_h100_8gpu.yaml**: YARN RoPE with 8-GPU sequence parallelism for maximum H100 utilization
- **needle_haystack_longrope.yaml**: LongRoPE scaling with custom short/long factors for extreme context lengths  
- **needle_haystack_llama3_rope.yaml**: Llama3 RoPE scaling optimized for Meta-Llama models

## Key Parameters

### RoPE Configuration
- `rope_scaling_type`: linear, dynamic, yarn, longrope, llama3
- `rope_scaling_factor`: Scaling factor for context extension
- `yarn_alpha`, `yarn_beta`: YARN-specific parameters
- `longrope_short_factor`, `longrope_long_factor`: LongRoPE interpolation factors

### Sequence Parallelism
- `sequence_parallel_size`: Number of GPUs (1-8)
- `sequence_parallel_mode`: zigzag-ring, ulysses, llama3

### Needle Haystack Settings
- `needle_context_lengths`: Token counts to evaluate
- `needle_depth_percents`: Needle positions (0-100%)
- `needle_save_inputs_outputs`: Save prompts and responses
- `needle_generation_temperature`: Sampling temperature (0.0 = deterministic)

## Output

Results saved to configured `save_dir`:
- `detailed_results.json`: Per-example results with scores
- `summary.json`: Aggregated statistics
- `inputs_outputs.json`: Input prompts and model outputs (if enabled)
- Visualization plots (if matplotlib available)