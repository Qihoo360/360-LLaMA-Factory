#!/bin/bash

# Quick Needle in Haystack (NIAH) Demo Setup Script
# Run this after cloning the repo to quickly test needle haystack evaluation

set -e

echo "Starting Quick Needle in Haystack Demo Setup..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/llamafactory" ]; then
    echo "Error: Please run this script from the 360-LLaMA-Factory root directory"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python -c "import llamafactory; print('LLamaFactory installed successfully')"

# Create a minimal demo config for quick testing
echo "Creating minimal demo configuration..."
cat > quick_niah_demo.yaml << 'EOF'
### model - Using a small model for quick testing
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0

### method
finetuning_type: full

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: alpaca
lang: en

### output
save_dir: demo_results

### eval
batch_size: 1

### needle haystack configuration - Quick demo with minimal examples
needle_context_lengths: [100, 200]  # Very short contexts for quick demo
needle_depth_percents: [0, 50, 100]  # Just 3 positions
needle_text: "The demo key is QUICK123."  # Demo needle
needle_question: "What is the demo key?"  # Demo question
EOF

echo "Demo configuration created: quick_niah_demo.yaml"

# Run the evaluation
echo "Running quick needle haystack evaluation demo..."
echo "This will test 2 context lengths × 3 positions = 6 examples"

python -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'quick_niah_demo.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"

echo ""
echo "Demo completed successfully!"
echo ""
echo "Results saved to: demo_results/"
echo "   - summary.json: Overall performance metrics"
echo "   - detailed_results.json: Individual example results" 
echo "   - needle_haystack_performance.png: Performance charts"
echo "   - needle_haystack_heatmap.png: Performance heatmap"
echo ""
echo "To customize the evaluation, edit the configuration parameters in:"
echo "   - quick_niah_demo.yaml (this demo)"
echo "   - needle_haystack_evaluation/needle_haystack_config.yaml (full config)"
echo ""
echo "For more information, see: needle_haystack_evaluation/README.md"