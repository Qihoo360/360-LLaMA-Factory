# Quick Start: Needle in Haystack Demo

This repository includes a **configurable Needle in Haystack evaluation** for testing long-context retrieval capabilities of language models.

## One-Command Demo

After cloning this repo to your Kubernetes pod (or any environment):

```bash
# Make the script executable and run
chmod +x quick_niah_demo.sh
./quick_niah_demo.sh
```

This will:
1. Install all dependencies
2. Create a minimal demo configuration
3. Run evaluation with 6 examples (2 context lengths × 3 positions)
4. Generate results and visualizations

## Expected Output

```
=== Needle in Haystack Evaluation Results ===
Overall Average Score: 1.000
Exact Match Rate: 1.000
Total Examples: 6

Scores by Context Length:
  ~100 tokens: 1.000
  ~200 tokens: 1.000

Scores by Needle Depth:
  0.0%: 1.000
  50.0%: 1.000
  100.0%: 1.000
```

## Generated Files

After running the demo, you'll find:
- `demo_results/summary.json` - Performance metrics
- `demo_results/detailed_results.json` - Individual results
- `demo_results/needle_haystack_performance.png` - Charts
- `demo_results/needle_haystack_heatmap.png` - Heatmap

## Customization

### Quick Demo Config (`quick_niah_demo.yaml`)
```yaml
needle_context_lengths: [100, 200]  # Short contexts for quick test
needle_depth_percents: [0, 50, 100]  # 3 positions
needle_text: "The demo key is QUICK123."
needle_question: "What is the demo key?"
```

### Full Config (`needle_haystack_evaluation/needle_haystack_config.yaml`)
```yaml
needle_context_lengths: [240, 480, 958, 1918]  # Full range
needle_depth_percents: [0, 25, 50, 75, 100]    # 5 positions
needle_text: "The secret key is 42 alpha bravo."
needle_question: "What is the secret key?"
```

## Custom Evaluation

To run with different parameters:

```bash
# Edit configuration
nano quick_niah_demo.yaml

# Run evaluation
python -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'quick_niah_demo.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

## Kubernetes Pod Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: niah-demo
spec:
  containers:
  - name: niah-container
    image: python:3.11
    command: ["/bin/bash"]
    args: ["-c", "git clone https://github.com/manncodes/360-LLaMA-Factory.git && cd 360-LLaMA-Factory && chmod +x quick_niah_demo.sh && ./quick_niah_demo.sh"]
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"
      limits:
        memory: "8Gi"
        cpu: "4"
```

## Full Documentation

For detailed configuration options and advanced usage:
- [Needle Haystack Evaluation README](needle_haystack_evaluation/README.md)
- [Configuration Examples](needle_haystack_evaluation/needle_haystack_config.yaml)

## What is Needle in Haystack?

Tests whether language models can accurately retrieve specific information (the "needle") from within large amounts of context text (the "haystack"). Critical for evaluating long-context understanding capabilities.

**Example:**
- **Context**: 1000 tokens of background text with hidden needle
- **Needle**: "The secret key is 42 alpha bravo."
- **Question**: "What is the secret key?"
- **Expected Answer**: "42 alpha bravo" or exact needle text