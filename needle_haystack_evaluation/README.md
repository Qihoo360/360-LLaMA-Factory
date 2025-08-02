# Needle in Haystack Evaluation

This directory contains the implementation and configuration for running "Needle in Haystack" evaluations using the LlamaFactory framework. This evaluation tests a model's ability to retrieve specific information from long contexts.

## Overview

The Needle in Haystack evaluation tests whether language models can accurately retrieve specific information (the "needle") from within large amounts of context text (the "haystack"). This is crucial for testing long-context understanding capabilities.

## Quick Start

### Method 1: Using the Python Script (Recommended)
```bash
cd /path/to/360-LLaMA-Factory
python3 run_needle_eval.py
```

### Method 2: Using Shell Script
```bash
cd /path/to/360-LLaMA-Factory
bash needle_haystack_evaluation/run_evaluation.sh
```

### Method 3: Direct CLI Execution
```bash
cd /path/to/360-LLaMA-Factory
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.argv = ['eval', 'needle_haystack_evaluation/needle_haystack_config.yaml']
from llamafactory.eval.evaluator import run_eval
run_eval()
"
```

## Configuration

The main configuration file is `needle_haystack_config.yaml`:

```yaml
### model
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0

### method
finetuning_type: full

### dataset
task: needle_haystack_proper
task_dir: evaluation
template: alpaca
lang: en

### output
save_dir: saves/tinyllama/needle_haystack_eval

### eval
batch_size: 1

### needle haystack configuration
needle_context_lengths: [240, 480, 958, 1918]  # Context lengths in tokens
needle_depth_percents: [0, 25, 50, 75, 100]    # Needle positions as percentages
needle_text: "The secret key is 42 alpha bravo."  # Text to find in context
needle_question: "What is the secret key?"      # Question about the needle
```

### Configuration Parameters

#### Basic Parameters
- **model_name_or_path**: HuggingFace model identifier or local path
- **finetuning_type**: Must be `full` for evaluation (required for proper template encoding)
- **task**: Must be `needle_haystack_proper` (uses the fixed dataset implementation)
- **task_dir**: Directory containing the evaluation dataset (`evaluation`)
- **template**: Template format for the model (e.g., `alpaca`, `llama3`, `vicuna`)
- **save_dir**: Output directory for results and visualizations
- **batch_size**: Number of examples to process simultaneously (recommend 1 for long contexts)

#### Needle Haystack Specific Parameters
- **needle_context_lengths**: List of context lengths in tokens to test (default: [250, 500, 1000, 2000])
- **needle_depth_percents**: List of needle position percentages in context (default: [0, 25, 50, 75, 100])
- **needle_text**: Custom text to hide in the context (default: "The secret key is 42 alpha bravo.")
- **needle_question**: Question to ask about the needle (default: "What is the secret key?")

## Dataset Details

The evaluation uses a custom dataset with configurable characteristics:

- **Context Lengths**: Configurable list of token counts (default: [240, 480, 958, 1,918] tokens)
- **Needle Positions**: Configurable depth percentages in context (default: [0%, 25%, 50%, 75%, 100%])
- **Needle Text**: Configurable text to find (default: "The secret key is 42 alpha bravo.")
- **Question**: Configurable retrieval question (default: "What is the secret key?")
- **Total Examples**: Varies based on configuration (default: 20 examples = 4 lengths × 5 positions)

### Example Custom Configuration

You can customize the evaluation by modifying the YAML parameters:

```yaml
# Test shorter contexts with more positions
needle_context_lengths: [100, 200, 400]
needle_depth_percents: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
needle_text: "The magic number is 1337."
needle_question: "What is the magic number?"
```

This would generate 33 examples (3 lengths × 11 positions) with a different needle.

## Output and Results

### Result Files

The evaluation generates several output files in the specified `save_dir`:

1. **summary.json**: Overall performance metrics
2. **detailed_results.json**: Individual example results with scores
3. **needle_haystack_performance.png**: Performance charts by context length and needle position
4. **needle_haystack_heatmap.png**: Performance heatmap visualization

### Performance Metrics

- **Overall Average Score**: Weighted average across all examples
- **Exact Match Rate**: Percentage of perfect needle retrievals
- **Scores by Context Length**: Performance breakdown by token count
- **Scores by Needle Depth**: Performance breakdown by position in context

### Example Output
```
=== Needle in Haystack Evaluation Results ===
Overall Average Score: 0.988
Exact Match Rate: 0.950
Total Examples: 20

Scores by Context Length:
  240 tokens: 1.000
  480 tokens: 0.950
  958 tokens: 1.000
  1,918 tokens: 1.000

Scores by Needle Depth:
  0.0%: 1.000
  25.0%: 0.938
  50.0%: 1.000
  75.0%: 1.000
  100.0%: 1.000
```

## Files in This Directory

- **needle_haystack_config.yaml**: Main configuration file
- **run_evaluation.sh**: Shell script for easy execution
- **README.md**: This documentation file

## Technical Implementation

### Custom Evaluator

The evaluation uses a custom evaluator (`src/llamafactory/eval/needle_haystack_evaluator.py`) that:

1. Loads the needle haystack dataset
2. Formats prompts using LlamaFactory's template system
3. Generates responses using the specified model
4. Evaluates needle retrieval accuracy
5. Creates visualizations and saves results

### Dataset Implementation

The dataset is implemented in `evaluation/needle_haystack/needle_haystack.py` and includes:

- Proper token-based context length measurement
- Configurable needle text and positions
- Integration with HuggingFace datasets API
- Fixed template encoding for reliable evaluation

## Troubleshooting

### Common Issues

1. **Empty Responses/0% Accuracy**: 
   - Ensure `finetuning_type: full` in config
   - Verify template compatibility with your model
   - Check that the dataset loads correctly

2. **Memory Issues**:
   - Reduce `batch_size` to 1
   - Use smaller models or shorter context lengths
   - Enable gradient checkpointing if available

3. **Template Errors**:
   - Try different templates: `alpaca`, `llama3`, `vicuna`, `chatml`
   - Ensure template matches your model's training format

### Performance Expectations

- **TinyLlama-1.1B**: ~98.8% accuracy (as tested)
- **Larger Models**: Generally better performance on longer contexts
- **Context Length Impact**: Performance may degrade with very long contexts (>2K tokens)

## Integration with LlamaFactory

This evaluation is fully integrated with the LlamaFactory ecosystem:

- Uses LlamaFactory's model loading and template system
- Compatible with LoRA adapters and quantization
- Supports all LlamaFactory model types (base, chat, instruct)
- Follows LlamaFactory's configuration conventions

## Citation

If you use this evaluation in your research, please cite:

```bibtex
@misc{needle_haystack_llamafactory,
  title={Needle in Haystack Evaluation for LlamaFactory},
  author={360-LLaMA-Factory Contributors},
  year={2025},
  url={https://github.com/Qihoo360/360-LLaMA-Factory}
}
```

Original Needle in Haystack concept by:
```bibtex
@misc{kamradt2023needle,
  title={LLMTest Needle In A Haystack},
  author={Greg Kamradt},
  year={2023},
  url={https://github.com/gkamradt/LLMTest_NeedleInAHaystack}
}
```