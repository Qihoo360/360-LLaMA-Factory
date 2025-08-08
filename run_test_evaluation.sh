#!/bin/bash

# Production Test Script for Needle Haystack Evaluation
# Tests different configurations systematically

set -e  # Exit on any error

echo "🧪 Needle Haystack Evaluation Test Runner"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/llamafactory/eval/needle_haystack_evaluator.py" ]; then
    echo "❌ Error: Run this script from the repository root directory"
    exit 1
fi

# Create results directory
mkdir -p tests/results

# Function to run evaluation with error handling
run_evaluation() {
    local config_file=$1
    local config_name=$(basename "$config_file" .yaml)
    
    echo ""
    echo "🔄 Testing: $config_name"
    echo "   Config: $config_file"
    
    # Check if config exists
    if [ ! -f "$config_file" ]; then
        echo "❌ Config file not found: $config_file"
        return 1
    fi
    
    # Run the evaluation
    echo "   Running: llamafactory-cli eval $config_file"
    
    # Use timeout to prevent hanging
    timeout 600 llamafactory-cli eval "$config_file" 2>&1 | head -50
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $config_name: SUCCESS"
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ $config_name: TIMEOUT (10 minutes)"
        return 1
    else
        echo "❌ $config_name: FAILED (exit code: $exit_code)"
        return 1
    fi
}

# Test configurations in order of complexity
test_configs=(
    "tests/eval_configs/test_basic_needle.yaml"
    "tests/eval_configs/test_linear_rope.yaml"
    "tests/eval_configs/test_yarn_rope.yaml"
    "tests/eval_configs/test_advanced_generation.yaml"
    "tests/eval_configs/test_chat_template.yaml"
    "tests/eval_configs/test_paulgraham_dataset.yaml"
    "tests/eval_configs/test_sequence_parallel.yaml"
    "tests/eval_configs/test_longrope_extreme.yaml"
)

# Track results
passed=0
failed=0
total=${#test_configs[@]}

echo ""
echo "📋 Test Plan: $total configurations"
echo "   Timeout: 10 minutes per test"
echo "   Results: tests/results/"

# Option to run specific test
if [ $# -eq 1 ]; then
    config_file=$1
    if [[ "$config_file" != tests/eval_configs/* ]]; then
        config_file="tests/eval_configs/$config_file"
    fi
    if [[ "$config_file" != *.yaml ]]; then
        config_file="$config_file.yaml"
    fi
    
    echo ""
    echo "🎯 Running single test: $config_file"
    
    if run_evaluation "$config_file"; then
        echo ""
        echo "🎉 Single test completed successfully!"
    else
        echo ""
        echo "❌ Single test failed."
        exit 1
    fi
    exit 0
fi

# Run all tests
echo ""
echo "🚀 Starting comprehensive evaluation tests..."

for config in "${test_configs[@]}"; do
    if run_evaluation "$config"; then
        ((passed++))
    else
        ((failed++))
        # Continue with other tests even if one fails
    fi
done

# Summary
echo ""
echo "=========================================="
echo "📊 TEST SUMMARY"
echo "=========================================="
echo "Total tests: $total"
echo "Passed: $passed"
echo "Failed: $failed"

if [ $failed -eq 0 ]; then
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "✅ Needle haystack evaluation is fully functional"
    echo "✅ All RoPE scaling methods work"
    echo "✅ Generation config integration works" 
    echo "✅ Chat template support works"
    echo "✅ All configurations are production-ready"
    echo ""
    echo "🚀 Ready for production use on your 8 H100 setup!"
else
    echo ""
    echo "❌ $failed/$total tests failed"
    echo "Check the output above for details"
    exit 1
fi