# Needle Haystack Evaluation - Usage Guide

## Quick Fix for Your Setup

Based on your error, here are the working configurations:

## ✅ **Single GPU Evaluation** (Recommended to start)

```bash
# Use your local model without sequence parallelism
llamafactory-cli eval eval_configs/needle_haystack_use_model_rope.yaml

# Or use the local model config
llamafactory-cli eval eval_configs/needle_haystack_local_model.yaml

# Updated H100 config (single GPU mode)
llamafactory-cli eval eval_configs/needle_haystack_h100_8gpu.yaml
```

## 🔧 **Multi-GPU Evaluation** (Advanced)

For multi-GPU, you need to use `torchrun`:

```bash
# 8 GPU evaluation
torchrun --nproc_per_node=8 --master_port=29500 \
  -m llamafactory.cli eval eval_configs/needle_haystack_multi_gpu.yaml

# 4 GPU evaluation  
torchrun --nproc_per_node=4 --master_port=29500 \
  -m llamafactory.cli eval eval_configs/needle_haystack_multi_gpu.yaml
```

## 📁 **Available Configurations**

1. **`needle_haystack_use_model_rope.yaml`** ✅ **RECOMMENDED**
   - Uses your local model `/exp/austin/F1_V4`
   - Leverages model's built-in RoPE scaling (8x factor, llama3 type)
   - Single GPU, no distributed setup needed
   - Tests up to 64K context length

2. **`needle_haystack_local_model.yaml`** ✅ **SIMPLE**
   - Basic configuration with your local model
   - Good for initial testing

3. **`needle_haystack_h100_8gpu.yaml`** ✅ **UPDATED**
   - Updated to use your local model
   - Single GPU mode by default
   - Uncomment sequence_parallel lines for multi-GPU

4. **`needle_haystack_multi_gpu.yaml`** 🚀 **ADVANCED**
   - For 8-GPU distributed evaluation
   - Use with `torchrun`
   - Tests extreme context lengths (up to 128K)

## 🎯 **Your Model Analysis**

Your model `/exp/austin/F1_V4` has excellent specs:
- **Max Position**: 131,072 tokens (131K context!)
- **RoPE Scaling**: Already configured (8x factor, llama3 type)
- **Architecture**: LlamaForCausalLM (40 layers, 4096 hidden)
- **Precision**: bfloat16

## 🚀 **Start Here**

```bash
# 1. Quick test with model's built-in RoPE
llamafactory-cli eval eval_configs/needle_haystack_use_model_rope.yaml

# 2. Check results
ls -la results/needle_haystack_model_rope/

# 3. View results
cat results/needle_haystack_model_rope/summary.json
```

## 🔧 **Troubleshooting**

- **403 Forbidden**: The configs now use your local model `/exp/austin/F1_V4`
- **AssertionError (distributed)**: Use single GPU configs or `torchrun` for multi-GPU
- **Out of Memory**: Reduce `needle_context_lengths` or `batch_size`

## 📊 **Expected Output**

The evaluation will create:
- `detailed_results.json` - Per-example results
- `summary.json` - Aggregated statistics  
- `inputs_outputs.json` - Full prompts and responses
- Performance visualizations (if matplotlib available)

Your model should perform very well given its 131K context window and 8x RoPE scaling!