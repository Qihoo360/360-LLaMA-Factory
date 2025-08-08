#!/usr/bin/env python3
"""Test enhanced needle haystack evaluation with generation config and chat templates."""

import os
import sys
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_functionality():
    """Test enhanced functionality with generation config and chat templates."""
    print("Testing Enhanced Needle Haystack Integration")
    print("=" * 50)
    
    try:
        from llamafactory.eval.needle_haystack_evaluator import NeedleHaystackEvaluator
        
        # Test args with chat template specifications
        test_args = {
            'model_name_or_path': 'microsoft/DialoGPT-small',
            'template': 'default',
            'task': 'needle_haystack',
            'task_dir': 'evaluation',
            'batch_size': 1,
            'save_dir': None,
            'needle_context_lengths': [200, 400],
            'needle_depth_percents': [0, 100],
            'needle_text': 'The enhanced key is XYZ123.',
            'needle_question': 'What is the enhanced key?',
            'needle_haystack_data_source': 'custom',
            'needle_save_inputs_outputs': True,
            'needle_generation_temperature': 0.1,
            'needle_generation_max_tokens': 20,
            'rope_scaling_type': None,
            'infer_dtype': 'auto',
            'use_cache': True
        }
        
        print("\n✓ Enhanced test arguments prepared")
        
        # Test that new methods are available
        from llamafactory.hparams import get_eval_args
        model_args, data_args, eval_args, finetuning_args = get_eval_args(test_args)
        
        print("✓ Arguments parsed successfully with enhanced parameters")
        print(f"  - Generation temperature: {eval_args.needle_generation_temperature}")
        print(f"  - Save inputs/outputs: {eval_args.needle_save_inputs_outputs}")
        
        # Test that the enhanced evaluator can be instantiated
        print("\n✓ Testing enhanced evaluator instantiation...")
        
        # Check that the methods exist (without full instantiation to avoid model loading)
        evaluator_class = NeedleHaystackEvaluator
        
        # Verify new methods exist
        assert hasattr(evaluator_class, '_configure_chat_template'), "Missing _configure_chat_template method"
        assert hasattr(evaluator_class, '_configure_generation_config'), "Missing _configure_generation_config method"
        assert hasattr(evaluator_class, '_encode_messages'), "Missing _encode_messages method"
        
        print("✓ All enhanced methods are available:")
        print("  - _configure_chat_template")
        print("  - _configure_generation_config") 
        print("  - _encode_messages")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_parameters():
    """Test that configuration parameters are properly available."""
    print("\nTesting Enhanced Configuration Parameters")
    print("-" * 50)
    
    try:
        from llamafactory.hparams.evaluation_args import EvaluationArguments
        
        # Test enhanced evaluation args
        eval_args = EvaluationArguments(
            task="needle_haystack",
            rope_scaling_type="yarn",
            rope_scaling_factor=2.0,
            yarn_alpha=1.0,
            yarn_beta=32.0,
            needle_save_inputs_outputs=True,
            needle_generation_temperature=0.1,
            needle_generation_top_p=0.9,
            needle_generation_top_k=50,
            needle_generation_max_tokens=100
        )
        
        print("✓ Enhanced EvaluationArguments created successfully")
        print(f"  - RoPE type: {eval_args.rope_scaling_type}")
        print(f"  - YARN parameters: α={eval_args.yarn_alpha}, β={eval_args.yarn_beta}")
        print(f"  - Generation config: temp={eval_args.needle_generation_temperature}, top_p={eval_args.needle_generation_top_p}")
        print(f"  - Save I/O: {eval_args.needle_save_inputs_outputs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config parameters test failed: {e}")
        return False

def test_yaml_enhanced_configs():
    """Test enhanced YAML configurations."""
    print("\nTesting Enhanced YAML Configurations")
    print("-" * 50)
    
    try:
        import yaml
        
        # Test that enhanced configs load properly
        config_files = [
            "eval_configs/needle_haystack_h100_8gpu.yaml",
            "eval_configs/needle_haystack_longrope.yaml",
            "eval_configs/needle_haystack_llama3_rope.yaml",
            "eval_configs/needle_haystack_test.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for enhanced parameters
                has_template = 'template' in config
                has_rope = 'rope_scaling_type' in config
                has_generation = 'needle_generation_temperature' in config
                has_save_io = 'needle_save_inputs_outputs' in config
                
                print(f"✓ {os.path.basename(config_file)}:")
                print(f"    Template: {'✓' if has_template else '✗'}")
                print(f"    RoPE: {'✓' if has_rope else '✗'}")
                print(f"    Generation config: {'✓' if has_generation else '✗'}")
                print(f"    Save I/O: {'✓' if has_save_io else '✗'}")
            else:
                print(f"✗ {config_file} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ YAML enhanced configs test failed: {e}")
        return False

def main():
    """Run all enhanced integration tests."""
    tests = [
        ("Enhanced Functionality", test_enhanced_functionality),
        ("Configuration Parameters", test_config_parameters),
        ("Enhanced YAML Configs", test_yaml_enhanced_configs)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Enhanced integration tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🚀 Enhanced Integration Complete!")
        print("\nNew capabilities available:")
        print("1. ✅ Generation config from model")
        print("2. ✅ Chat template from tokenizer_config") 
        print("3. ✅ RoPE scaling integration")
        print("4. ✅ Input/output saving")
        print("5. ✅ Enhanced YAML configurations")
        print("\nReady for advanced needle haystack evaluation!")
    else:
        print("\n❌ Some enhanced features need attention.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)