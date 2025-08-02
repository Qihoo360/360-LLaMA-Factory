#!/bin/bash
# Needle in Haystack Evaluation Runner
# This script runs the complete needle haystack evaluation with visualization

echo "Starting Needle in Haystack Evaluation"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "unit_test_needle_haystack.yaml" ]; then
    echo "Error: Please run this script from the 360-LLaMA-Factory root directory"
    exit 1
fi

# Create results directory
mkdir -p results/needle_haystack_evaluation

echo "Configuration:"
echo "  Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "  Dataset: needle_haystack_proper (fixed implementation)"
echo "  Template: alpaca (fixed encoding)"  
echo "  Context lengths: [250, 500, 1000, 2000] tokens"
echo "  Output: results/needle_haystack_evaluation/"
echo ""

# Run the evaluation
echo "Running evaluation through llamafactory-cli..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from llamafactory.cli import main
sys.argv = [
    'llamafactory-cli', 'eval',
    '--model_name_or_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    '--task', 'needle_haystack_proper',
    '--task_dir', 'evaluation', 
    '--template', 'alpaca',
    '--batch_size', '1',
    '--save_dir', 'results/needle_haystack_evaluation',
    '--lang', 'en'
]
main()
"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: results/needle_haystack_evaluation/"
    echo "Generated files:"
    echo "  - detailed_results.json     (individual test results)"
    echo "  - summary.json             (aggregated statistics)"
    echo "  - needle_haystack_performance.png (performance charts)"
    echo "  - needle_haystack_heatmap.png     (performance heatmap)"
    echo ""
    echo "View visualizations:"
    echo "  - Performance charts: results/needle_haystack_evaluation/needle_haystack_performance.png"
    echo "  - Performance heatmap: results/needle_haystack_evaluation/needle_haystack_heatmap.png"
else
    echo ""
    echo "Evaluation failed. Check the error messages above."
    exit 1
fi