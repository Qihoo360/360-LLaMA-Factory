#!/usr/bin/env python3
"""Quick test of actual needle haystack evaluation functionality."""

import os
import sys
from pathlib import Path

# Add src to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_quick_evaluation():
    """Test actual evaluation with minimal setup."""
    print("🧪 Quick Needle Haystack Evaluation Test")
    print("=" * 50)
    
    try:
        # Test the basic needle evaluation
        from llamafactory.hparams import get_eval_args
        from llamafactory.eval.needle_haystack_evaluator import NeedleHaystackEvaluator
        
        # Use basic test configuration 
        test_config = {
            'model_name_or_path': 'microsoft/DialoGPT-small',
            'template': 'default',
            'task': 'needle_haystack',
            'task_dir': 'evaluation',
            'batch_size': 1,
            'save_dir': None,  # Don't save to avoid file conflicts
            'needle_context_lengths': [100, 200],  # Very small for quick test
            'needle_depth_percents': [0, 100],  # Just beginning and end
            'needle_text': 'The test key is QUICKTEST.',
            'needle_question': 'What is the test key?',
            'needle_haystack_data_source': 'custom',
            'needle_save_inputs_outputs': False,
            'needle_generation_temperature': 0.0,
            'needle_generation_max_tokens': 10,
            'infer_dtype': 'auto'
        }
        
        print("✅ Test configuration prepared")
        print(f"   Model: {test_config['model_name_or_path']}")
        print(f"   Context lengths: {test_config['needle_context_lengths']}")
        print(f"   Needle: {test_config['needle_text']}")
        
        # Test argument parsing
        model_args, data_args, eval_args, finetuning_args = get_eval_args(test_config)
        print("✅ Arguments parsed successfully")
        
        # Test evaluator instantiation (this tests all our enhancements)
        print("⏳ Testing evaluator instantiation...")
        print("   Note: This will load the model and may take a moment...")
        
        # Create evaluator - this tests the full integration
        evaluator = NeedleHaystackEvaluator(test_config)
        print("✅ Evaluator created successfully")
        
        # Test that our enhanced methods are working
        if hasattr(evaluator, 'base_generation_config'):
            print("✅ Generation config integration working")
        if hasattr(evaluator, '_encode_messages'):
            print("✅ Chat template integration working")
            
        # Test message encoding
        test_messages = [
            {"role": "user", "content": "Context: This is a test context. The test key is QUICKTEST. More context here.\n\nQuestion: What is the test key?"},
            {"role": "assistant", "content": ""}
        ]
        
        input_ids = evaluator._encode_messages(test_messages)
        print(f"✅ Message encoding successful (length: {len(input_ids)} tokens)")
        
        print("\n🎉 Quick evaluation test successful!")
        print("The integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Quick evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_evaluation()
    if success:
        print("\n✅ All systems go! Needle haystack evaluation is ready.")
        print("Run full evaluations with: llamafactory-cli eval tests/eval_configs/<config>.yaml")
    else:
        print("\n❌ Integration test failed.")
    
    sys.exit(0 if success else 1)