#!/bin/bash

# Multi-GPU Needle Haystack Evaluation Script
# This script properly initializes distributed training for sequence parallelism

set -e

# Default values
CONFIG_FILE=""
GPUS=8
PORT=29500
MASTER_ADDR="localhost"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2" 
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --config <config.yaml> [--gpus <num_gpus>] [--port <port>] [--master_addr <addr>]"
            echo ""
            echo "Examples:"
            echo "  # 8 GPU evaluation"
            echo "  $0 --config eval_configs/needle_haystack_8gpu_distributed.yaml --gpus 8"
            echo ""
            echo "  # 4 GPU evaluation"  
            echo "  $0 --config eval_configs/needle_haystack_4gpu_distributed.yaml --gpus 4"
            echo ""
            echo "  # Multi-node evaluation"
            echo "  $0 --config eval_configs/needle_haystack_8gpu_distributed.yaml --gpus 8 --master_addr <master_node_ip>"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

# Display configuration
echo "🚀 Starting Multi-GPU Needle Haystack Evaluation"
echo "Configuration: $CONFIG_FILE"
echo "GPUs: $GPUS"
echo "Master Address: $MASTER_ADDR"
echo "Port: $PORT"
echo ""

# Check for required dependencies
echo "🔍 Checking dependencies..."
python3 -c "import ring_flash_attn; print('✅ ring_flash_attn installed')" 2>/dev/null || {
    echo "❌ ring_flash_attn not installed. Install with: pip install ring_flash_attn"
    exit 1
}

python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || {
    echo "❌ PyTorch not installed properly"
    exit 1
}

echo "✅ All dependencies check passed"
echo ""

# Run distributed evaluation
echo "🔥 Launching distributed evaluation..."
echo "Command: torchrun --nproc_per_node=$GPUS --master_addr=$MASTER_ADDR --master_port=$PORT -m llamafactory.cli eval $CONFIG_FILE"
echo ""

torchrun \
    --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    -m llamafactory.cli eval "$CONFIG_FILE"

echo ""
echo "🎉 Evaluation completed! Check the results directory specified in your config."