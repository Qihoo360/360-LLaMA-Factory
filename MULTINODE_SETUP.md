# Multi-GPU Sequence Parallelism Setup

## 🚨 Important: Sequence Parallelism Requires Proper Distributed Setup

**The AssertionError you encountered occurs because sequence parallelism requires distributed initialization with `torchrun`.**

## ✅ Correct Usage Commands

### 8 GPU Evaluation (Single Node)
```bash
# Option 1: Using the launch script (recommended)
./run_distributed_eval.sh --config eval_configs/needle_haystack_8gpu_distributed.yaml --gpus 8

# Option 2: Direct torchrun command
torchrun --nproc_per_node=8 --master_port=29500 \
  -m llamafactory.cli eval eval_configs/needle_haystack_8gpu_distributed.yaml
```

### 4 GPU Evaluation
```bash
./run_distributed_eval.sh --config eval_configs/needle_haystack_4gpu_distributed.yaml --gpus 4

# Or directly:
torchrun --nproc_per_node=4 --master_port=29500 \
  -m llamafactory.cli eval eval_configs/needle_haystack_4gpu_distributed.yaml
```

### Ulysses Sequence Parallelism (8 GPU)
```bash
./run_distributed_eval.sh --config eval_configs/needle_haystack_8gpu_ulysses.yaml --gpus 8
```

## 🚫 What NOT To Do

**❌ This will fail with AssertionError:**
```bash
# DON'T do this - sequence parallelism needs distributed setup
llamafactory-cli eval eval_configs/needle_haystack_8gpu_distributed.yaml
```

## 🔧 Available Configurations

### Single GPU Configurations
- `needle_haystack_h100_8gpu.yaml` - Single GPU, no sequence parallelism
- `needle_haystack_use_model_rope.yaml` - Single GPU with model's RoPE
- `needle_haystack_local_model.yaml` - Basic single GPU

### Multi-GPU Distributed Configurations  
- `needle_haystack_8gpu_distributed.yaml` - 8 GPU zigzag-ring sequence parallelism
- `needle_haystack_4gpu_distributed.yaml` - 4 GPU zigzag-ring sequence parallelism
- `needle_haystack_8gpu_ulysses.yaml` - 8 GPU Ulysses sequence parallelism

## 🌐 Multi-Node Setup (Multiple Machines)

### Master Node (Machine 1):
```bash
# Replace <master_node_ip> with actual IP address
./run_distributed_eval.sh \
  --config eval_configs/needle_haystack_8gpu_distributed.yaml \
  --gpus 8 \
  --master_addr <master_node_ip> \
  --port 29500
```

### Worker Nodes (Other Machines):
```bash
# Set NODE_RANK for each additional node (1, 2, 3, ...)
export NODE_RANK=1  # Increment for each worker node

torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr=<master_node_ip> \
  --master_port=29500 \
  -m llamafactory.cli eval eval_configs/needle_haystack_8gpu_distributed.yaml
```

## 📋 Prerequisites

### Required Dependencies:
```bash
pip install ring_flash_attn  # For zigzag-ring mode
pip install flash-attn       # For flash attention
```

### Hardware Requirements:
- Multiple GPUs (2, 4, 8, etc.)
- Sufficient GPU memory for your context lengths
- Fast interconnect (NVLink/InfiniBand recommended)

## 🎯 Testing Your Setup

### Quick Test (4 GPU):
```bash
# Test with smaller context lengths first
./run_distributed_eval.sh --config eval_configs/needle_haystack_4gpu_distributed.yaml --gpus 4
```

### Full Scale Test (8 GPU):
```bash
# Full evaluation with extreme context lengths
./run_distributed_eval.sh --config eval_configs/needle_haystack_8gpu_distributed.yaml --gpus 8
```

## 🔍 Troubleshooting

### Common Issues:

1. **AssertionError: `assert dist.is_initialized()`**
   - Solution: Use `torchrun` or the launch script, not `llamafactory-cli` directly

2. **`ring_flash_attn` not found**
   - Solution: `pip install ring_flash_attn`

3. **CUDA out of memory**
   - Solution: Reduce `needle_context_lengths` or increase `sequence_parallel_size`

4. **Port already in use**
   - Solution: Change `--port` to different value (e.g., 29501, 29502)

## 🚀 Expected Performance

With proper multi-GPU setup, you should see:
- Linear scaling with more GPUs for long sequences
- Ability to handle 128K+ context lengths
- Efficient memory utilization across GPUs
- Faster evaluation compared to single GPU

Your model `/exp/austin/F1_V4` with 131K max position embeddings is perfect for testing extreme context lengths with sequence parallelism!