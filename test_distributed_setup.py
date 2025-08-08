#!/usr/bin/env python3
"""Test script to validate distributed setup for sequence parallelism."""

import os
import sys
import torch
import torch.distributed as dist

def test_distributed_setup():
    """Test if distributed environment is properly set up."""
    print("🔍 Testing Distributed Setup")
    print("=" * 50)
    
    # Check if we're in a distributed environment
    if not dist.is_available():
        print("❌ PyTorch distributed is not available")
        return False
    
    # Check if distributed is initialized
    if not dist.is_initialized():
        print("❌ Distributed backend is not initialized")
        print("💡 This script must be run with torchrun:")
        print("   torchrun --nproc_per_node=8 test_distributed_setup.py")
        return False
    
    # Get distributed info
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"✅ Distributed backend initialized: {dist.get_backend()}")
    print(f"✅ World size: {world_size}")
    print(f"✅ Global rank: {rank}")
    print(f"✅ Local rank: {local_rank}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    cuda_device_count = torch.cuda.device_count()
    print(f"✅ CUDA devices available: {cuda_device_count}")
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    print(f"✅ Using CUDA device: {device}")
    
    # Test basic distributed operation
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size))
    
    if tensor.item() == expected_sum:
        print(f"✅ Distributed communication test passed: {tensor.item()} == {expected_sum}")
    else:
        print(f"❌ Distributed communication test failed: {tensor.item()} != {expected_sum}")
        return False
    
    # Test sequence parallelism requirements
    print(f"\n🧪 Testing Sequence Parallelism Requirements")
    
    # Check if world_size is compatible with common SP sizes
    sp_sizes = [2, 4, 8]
    compatible_sp_sizes = [sp for sp in sp_sizes if world_size % sp == 0]
    
    if compatible_sp_sizes:
        print(f"✅ Compatible sequence_parallel_sizes: {compatible_sp_sizes}")
    else:
        print(f"⚠️ World size {world_size} not compatible with common SP sizes {sp_sizes}")
    
    # Check for ring_flash_attn
    try:
        import ring_flash_attn
        print("✅ ring_flash_attn is installed (required for zigzag-ring mode)")
    except ImportError:
        print("❌ ring_flash_attn not installed (required for zigzag-ring mode)")
        print("   Install with: pip install ring_flash_attn")
    
    # Check for flash_attn
    try:
        import flash_attn
        print("✅ flash_attn is installed")
    except ImportError:
        print("⚠️ flash_attn not installed (recommended)")
    
    print(f"\n🎉 Rank {rank}: All tests passed! Ready for sequence parallelism.")
    return True

def main():
    """Main test function."""
    success = test_distributed_setup()
    
    # Only print summary from rank 0
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"\n{'='*50}")
        if success:
            print("🎉 Distributed setup validation PASSED!")
            print("✅ You can now run multi-GPU evaluations with:")
            print("   ./run_distributed_eval.sh --config eval_configs/needle_haystack_8gpu_distributed.yaml --gpus 8")
        else:
            print("❌ Distributed setup validation FAILED!")
            print("🔧 Please fix the issues above before running multi-GPU evaluations.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())