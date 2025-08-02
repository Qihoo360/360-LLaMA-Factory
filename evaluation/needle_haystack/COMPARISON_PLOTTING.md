# Needle in Haystack Comparison Plotting

This guide explains how to use the comparison plotting script to analyze and visualize needle haystack evaluation results across multiple experiments.

## Overview

The `plot_niah_comparison.py` script automatically discovers evaluation results and generates comprehensive comparison visualizations including:

- Overall performance comparison across experiments
- Performance analysis by context length
- Performance analysis by needle depth position
- Detailed heatmaps for individual experiments
- Comprehensive comparison reports

## Usage

### Basic Usage

Generate all comparison plots for all discovered experiments:

```bash
python plot_niah_comparison.py --plot-type all --output-dir comparison_plots
```

### Specific Plot Types

Generate only overall comparison:
```bash
python plot_niah_comparison.py --plot-type overall
```

Generate context length analysis:
```bash
python plot_niah_comparison.py --plot-type context
```

Generate depth position analysis:
```bash
python plot_niah_comparison.py --plot-type depth
```

Generate heatmap for specific experiment:
```bash
python plot_niah_comparison.py --plot-type heatmap --experiment-for-heatmap tinyllama_needle_haystack_eval
```

### Compare Specific Experiments

Compare only selected experiments:
```bash
python plot_niah_comparison.py --experiments tinyllama_needle_haystack_eval tinyllama_needle_haystack_paulgraham --plot-type all --output-dir selected_comparison
```

### Custom Saves Directory

Use a different saves directory:
```bash
python plot_niah_comparison.py --saves-dir custom_saves --plot-type all
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--saves-dir` | Directory containing saved results | `saves` |
| `--experiments` | Specific experiments to compare | All discovered |
| `--output-dir` | Output directory for plots | `comparison_plots` |
| `--plot-type` | Type of plot to generate: `overall`, `context`, `depth`, `heatmap`, `all` | `all` |
| `--experiment-for-heatmap` | Specific experiment for heatmap (required if plot-type=heatmap) | None |

## Output Files

When using `--plot-type all`, the script generates:

### Visualization Files
- `overall_comparison.png` - Overall performance comparison across experiments
- `context_length_comparison.png` - Performance trends by context length
- `depth_comparison.png` - Performance trends by needle depth position
- `heatmap_[experiment_name].png` - Individual heatmaps for each experiment

### Analysis Report
- `comparison_report.txt` - Comprehensive text report with:
  - Experiment summary table
  - Performance rankings
  - Configuration details

## Experiment Discovery

The script automatically discovers experiments by:

1. Scanning the saves directory for subdirectories
2. Looking for `summary.json` and `detailed_results.json` files
3. Extracting metadata from paths and result data
4. Identifying data sources (custom, paulgraham, directory)

## Data Source Detection

The script automatically identifies data sources based on:
- Path patterns (`paulgraham`, `custom`, `directory`)
- Needle text content (e.g., `PG42DEMO` indicates Paul Graham data)
- Default fallback to `custom`

## Example Output Structure

```
comparison_plots/
├── overall_comparison.png           # Overall performance comparison
├── context_length_comparison.png    # Performance by context length
├── depth_comparison.png            # Performance by needle depth
├── heatmap_experiment1.png         # Individual experiment heatmaps
├── heatmap_experiment2.png
└── comparison_report.txt           # Detailed text report
```

## Interpretation Guide

### Overall Comparison
- **Average Score**: Primary performance metric (0-1 scale)
- **Exact Match Rate**: Percentage of perfect matches
- **Total Examples**: Number of test cases
- **Data Source Analysis**: Performance differences across data sources

### Context Length Analysis
- Shows how performance degrades (or maintains) with longer contexts
- Useful for identifying model context length limitations
- Threshold lines at 90% and 80% performance

### Depth Analysis
- Shows how needle position affects retrieval accuracy
- Common patterns: beginning and end positions often perform better
- Middle positions can be more challenging

### Heatmaps
- Detailed view of performance across all context length × depth combinations
- Color coding: Green (good), Yellow (moderate), Red (poor)
- Helps identify specific problematic configurations

## Integration with Evaluation Workflow

1. **Run Evaluations**: Use the evaluation commands to generate results
2. **Compare Results**: Use this script to analyze and compare experiments
3. **Identify Issues**: Use visualizations to spot performance patterns
4. **Iterate**: Make model or configuration improvements based on insights

## Troubleshooting

### No Results Found
- Verify the saves directory path
- Ensure evaluations have completed successfully
- Check that `summary.json` and `detailed_results.json` exist

### Missing Dependencies
Install required packages:
```bash
pip install matplotlib seaborn pandas numpy
```

### Large Number of Experiments
For many experiments, consider:
- Using `--experiments` to select specific ones
- Filtering by data source or model type
- Generating reports in batches