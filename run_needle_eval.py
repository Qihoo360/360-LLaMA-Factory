#!/usr/bin/env python3
"""
Simple script to run needle haystack evaluation with YAML config
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set sys.argv for YAML parsing
if len(sys.argv) > 1:
    yaml_file = sys.argv[1]
else:
    yaml_file = 'needle_haystack_evaluation/needle_haystack_config.yaml'

sys.argv = ['eval', yaml_file]

# Import and run evaluation
from llamafactory.eval.evaluator import run_eval

if __name__ == "__main__":
    print(f"Running needle haystack evaluation with config: {yaml_file}")
    run_eval()