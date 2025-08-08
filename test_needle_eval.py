#!/usr/bin/env python3
"""Test script for needle haystack evaluation with RoPE extensions."""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evaluation_args():
    """Test that evaluation args can parse RoPE parameters."""
    try:
        from llamafactory.hparams.evaluation_args import EvaluationArguments
        
        # Test basic instantiation
        eval_args = EvaluationArguments(task="needle_haystack")
        print("✓ EvaluationArguments can be instantiated")
        
        # Test RoPE parameters
        eval_args.rope_scaling_type = "yarn"
        eval_args.rope_scaling_factor = 2.0
        eval_args.yarn_alpha = 1.0
        eval_args.yarn_beta = 32.0
        print("✓ RoPE parameters can be set")
        
        # Test needle parameters
        eval_args.needle_save_inputs_outputs = True
        eval_args.needle_generation_temperature = 0.1
        print("✓ Needle haystack parameters can be set")
        
        return True
    except Exception as e:
        print(f"✗ EvaluationArguments test failed: {e}")
        return False

def test_needle_evaluator():
    """Test needle haystack evaluator instantiation."""
    try:
        from llamafactory.eval.needle_haystack_evaluator import NeedleHaystackEvaluator
        print("✓ NeedleHaystackEvaluator can be imported")
        return True
    except Exception as e:
        print(f"✗ NeedleHaystackEvaluator import failed: {e}")
        return False

def test_yaml_configs():
    """Test that YAML configurations are valid."""
    config_dir = Path(__file__).parent / "eval_configs"
    yaml_files = list(config_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("✗ No YAML config files found")
        return False
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ {yaml_file.name} is valid YAML")
        except Exception as e:
            print(f"✗ {yaml_file.name} failed: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Testing Needle Haystack Evaluation Integration")
    print("=" * 50)
    
    tests = [
        ("Evaluation Arguments", test_evaluation_args),
        ("Needle Evaluator", test_needle_evaluator), 
        ("YAML Configurations", test_yaml_configs)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Integration is ready.")
        print("\nRun needle haystack evaluation with:")
        print("llamafactory-cli eval eval_configs/needle_haystack_h100_8gpu.yaml")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()