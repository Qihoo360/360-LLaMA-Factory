# Cache Troubleshooting for Needle Haystack Evaluation

## Problem: Configuration Changes Not Taking Effect

When you modify configuration parameters (like `needle_context_lengths`) in your YAML file, the evaluation might still use old cached values. This happens because HuggingFace datasets library caches generated datasets.

## Solutions

### Solution 1: Clear Cache Before Running (Recommended)

Run the cache clearing script before evaluation:

```bash
# Clear cache
./clear_needle_cache.sh

# Then run your evaluation
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','paulgraham_demo_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"
```

### Solution 2: Force Redownload Mode

Add force redownload to your YAML configuration:

```yaml
# Add this to your config file
download_mode: force_redownload
```

Or set it via environment variable:

```bash
export DOWNLOAD_MODE=force_redownload
python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','paulgraham_demo_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"
```

### Solution 3: Manual Cache Clearing

Clear HuggingFace cache manually:

```bash
# Clear all needle haystack caches
rm -rf ~/.cache/huggingface/datasets/*needle_haystack*
rm -rf ~/.cache/huggingface/modules/*needle_haystack*

# Or clear all HuggingFace datasets cache (more aggressive)
rm -rf ~/.cache/huggingface/datasets/*
```

### Solution 4: Use Different Save Directory

Change the `save_dir` in your config to force new evaluation:

```yaml
# Original
save_dir: saves/tinyllama/needle_haystack_paulgraham

# Changed (add version or timestamp)
save_dir: saves/tinyllama/needle_haystack_paulgraham_v2
# or
save_dir: saves/tinyllama/needle_haystack_paulgraham_20250802
```

## How to Verify Configuration is Applied

1. **Check the output** - The evaluation should show the correct number of examples:
   ```
   Processing examples: 100%|██████████| 15/15
   ```
   (15 = 3 context lengths × 5 positions)

2. **Check the summary** - Context lengths should match your configuration:
   ```
   Scores by Context Length:
     300 tokens: 1.000
     600 tokens: 1.000
     900 tokens: 1.000
   ```

3. **Check detailed results** - Look at `detailed_results.json`:
   ```json
   {
     "context_length": 300,  // Should match your config
     ...
   }
   ```

## Prevention Tips

1. **Always clear cache when changing configuration parameters**:
   - Context lengths
   - Needle positions
   - Needle text
   - Data source

2. **Use unique save directories** for different experiments

3. **Add timestamp to save_dir** for reproducibility:
   ```yaml
   save_dir: saves/experiment_$(date +%Y%m%d_%H%M%S)
   ```

## Common Cache Locations

- **HuggingFace datasets**: `~/.cache/huggingface/datasets/`
- **HuggingFace modules**: `~/.cache/huggingface/modules/`
- **Custom cache**: `./cache/` (if set)
- **Model cache**: `~/.cache/huggingface/hub/`

## Debug Commands

Check what's in the cache:

```bash
# List cached datasets
ls -la ~/.cache/huggingface/datasets/

# Check specific needle haystack cache
ls -la ~/.cache/huggingface/datasets/*needle*

# Check cache size
du -sh ~/.cache/huggingface/
```

## Quick One-Liner with Cache Clear

```bash
# All-in-one: clear cache and run evaluation
rm -rf ~/.cache/huggingface/datasets/*needle* && python3 -c "import sys,os;sys.path.insert(0,'src');sys.argv=['e','paulgraham_demo_config.yaml'];from llamafactory.eval.evaluator import run_eval;run_eval()"
```

## Why This Happens

HuggingFace datasets uses a caching mechanism based on:
- Dataset name
- Script content hash
- Version

However, it doesn't automatically detect changes in dynamic configuration passed at runtime. Our implementation now includes configuration parameters in the cache key to force regeneration when settings change.