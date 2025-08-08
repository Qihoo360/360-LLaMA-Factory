#!/usr/bin/env python3
"""
Comprehensive test runner for all needle haystack evaluation configurations.
Tests all YAML configurations to verify functionality.
"""

import os
import sys
import yaml
import subprocess
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_yaml_config(yaml_path):
    """Load and validate YAML configuration."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config, None
    except Exception as e:
        return None, str(e)

def test_yaml_parsing():
    """Test that all YAML configs parse correctly."""
    print("=" * 60)
    print("YAML CONFIGURATION PARSING TESTS")
    print("=" * 60)
    
    test_dir = Path(__file__).parent / "eval_configs"
    yaml_files = list(test_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("❌ No YAML test files found!")
        return False
    
    all_passed = True
    configs = {}
    
    for yaml_file in sorted(yaml_files):
        config, error = load_yaml_config(yaml_file)
        if error:
            print(f"❌ {yaml_file.name}: Parse error - {error}")
            all_passed = False
        else:
            print(f"✅ {yaml_file.name}: Parse successful")
            configs[yaml_file.name] = config
    
    return all_passed, configs

def validate_config_completeness(configs):
    """Validate that configs have required parameters."""
    print("\n" + "=" * 60)
    print("CONFIGURATION COMPLETENESS VALIDATION")
    print("=" * 60)
    
    required_fields = [
        'model_name_or_path', 'template', 'task', 'task_dir',
        'needle_context_lengths', 'needle_depth_percents', 
        'needle_text', 'needle_question'
    ]
    
    all_passed = True
    
    for config_name, config in configs.items():
        missing_fields = []
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ {config_name}: Missing fields - {missing_fields}")
            all_passed = False
        else:
            print(f"✅ {config_name}: All required fields present")
    
    return all_passed

def test_parameter_variations(configs):
    """Test that configs cover different parameter variations."""
    print("\n" + "=" * 60)
    print("PARAMETER VARIATION COVERAGE")
    print("=" * 60)
    
    # Test coverage tracking
    rope_types = set()
    generation_temps = set()
    datasets = set()
    parallel_modes = set()
    
    for config_name, config in configs.items():
        # RoPE scaling types
        rope_type = config.get('rope_scaling_type', 'none')
        rope_types.add(rope_type)
        
        # Generation temperatures
        temp = config.get('needle_generation_temperature', 0.0)
        generation_temps.add(temp)
        
        # Data sources
        data_source = config.get('needle_haystack_data_source', 'custom')
        datasets.add(data_source)
        
        # Parallel modes
        parallel_mode = config.get('sequence_parallel_mode', 'none')
        parallel_modes.add(parallel_mode)
        
        print(f"📊 {config_name}:")
        print(f"    RoPE: {rope_type}, Temp: {temp}, Data: {data_source}, Parallel: {parallel_mode}")
    
    print(f"\n📈 Coverage Summary:")
    print(f"    RoPE types: {rope_types}")
    print(f"    Generation temps: {generation_temps}")
    print(f"    Datasets: {datasets}")
    print(f"    Parallel modes: {parallel_modes}")
    
    # Check coverage
    coverage_good = True
    if len(rope_types) < 4:  # Should have none, linear, yarn, longrope
        print(f"⚠️  Limited RoPE type coverage: {rope_types}")
        coverage_good = False
    if len(generation_temps) < 3:  # Should have deterministic and creative
        print(f"⚠️  Limited temperature coverage: {generation_temps}")
        coverage_good = False
        
    return coverage_good

def test_integration_functionality():
    """Test that the integration functionality works."""
    print("\n" + "=" * 60)
    print("INTEGRATION FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        # Test imports
        from llamafactory.hparams.evaluation_args import EvaluationArguments
        from llamafactory.eval.needle_haystack_evaluator import NeedleHaystackEvaluator
        from llamafactory.hparams import get_eval_args
        
        print("✅ All required modules import successfully")
        
        # Test evaluation args with enhanced parameters
        test_args = {
            'task': 'needle_haystack',
            'model_name_or_path': 'microsoft/DialoGPT-small',
            'template': 'default',
            'rope_scaling_type': 'yarn',
            'yarn_alpha': 1.0,
            'yarn_beta': 32.0,
            'needle_save_inputs_outputs': True
        }
        
        model_args, data_args, eval_args, finetuning_args = get_eval_args(test_args)
        print("✅ Enhanced evaluation arguments parse successfully")
        
        # Test that all enhanced methods are available
        evaluator_methods = [
            '_configure_chat_template',
            '_configure_generation_config', 
            '_encode_messages'
        ]
        
        for method in evaluator_methods:
            if hasattr(NeedleHaystackEvaluator, method):
                print(f"✅ Method '{method}' available")
            else:
                print(f"❌ Method '{method}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_quick_eval_test():
    """Run a quick evaluation test with basic config."""
    print("\n" + "=" * 60)  
    print("QUICK EVALUATION TEST")
    print("=" * 60)
    
    try:
        # Test with the basic config
        basic_config = Path(__file__).parent / "eval_configs" / "test_basic_needle.yaml"
        
        if not basic_config.exists():
            print("❌ Basic test config not found")
            return False
        
        print(f"🧪 Testing basic configuration: {basic_config.name}")
        print("Note: This will test argument parsing and setup, but not full model loading")
        
        # Test that the config would work with llamafactory-cli
        # We'll simulate the argument parsing without actually running the model
        import yaml
        with open(basic_config, 'r') as f:
            config = yaml.safe_load(f)
        
        from llamafactory.hparams import get_eval_args
        model_args, data_args, eval_args, finetuning_args = get_eval_args(config)
        
        print(f"✅ Configuration parsed successfully")
        print(f"    Model: {model_args.model_name_or_path}")
        print(f"    Task: {eval_args.task}")
        print(f"    Context lengths: {eval_args.needle_context_lengths}")
        print(f"    RoPE type: {getattr(eval_args, 'rope_scaling_type', 'none')}")
        print(f"    Save I/O: {eval_args.needle_save_inputs_outputs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick eval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all evaluation configuration tests."""
    print("🧪 COMPREHENSIVE NEEDLE HAYSTACK EVALUATION CONFIG TESTS")
    print("=" * 80)
    
    os.chdir(Path(__file__).parent.parent)  # Change to repo root
    
    test_results = []
    
    # Test 1: YAML parsing
    yaml_passed, configs = test_yaml_parsing()
    test_results.append(("YAML Parsing", yaml_passed))
    
    if not yaml_passed:
        print("\n❌ YAML parsing failed. Stopping tests.")
        return False
    
    # Test 2: Configuration completeness
    completeness_passed = validate_config_completeness(configs)
    test_results.append(("Configuration Completeness", completeness_passed))
    
    # Test 3: Parameter variation coverage
    coverage_passed = test_parameter_variations(configs)
    test_results.append(("Parameter Coverage", coverage_passed))
    
    # Test 4: Integration functionality
    integration_passed = test_integration_functionality()
    test_results.append(("Integration Functionality", integration_passed))
    
    # Test 5: Quick evaluation test
    eval_passed = run_quick_eval_test()
    test_results.append(("Quick Evaluation", eval_passed))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_count = 0
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if passed:
            passed_count += 1
    
    print(f"\nOverall: {passed_count}/{len(test_results)} tests passed")
    
    if passed_count == len(test_results):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nAll YAML configurations are ready for use:")
        print("📁 tests/eval_configs/")
        for config_file in sorted(Path("tests/eval_configs").glob("*.yaml")):
            print(f"   └── {config_file.name}")
        print("\nUsage: llamafactory-cli eval tests/eval_configs/<config_name>.yaml")
        return True
    else:
        print(f"\n❌ {len(test_results) - passed_count} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)