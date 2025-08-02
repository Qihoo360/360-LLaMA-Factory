#!/bin/bash

# Fix for NonMatchingSplitsSizeError in needle haystack evaluation

echo "Fixing needle haystack dataset cache issues..."

# Clear all needle haystack related caches
echo "1. Clearing HuggingFace datasets cache..."
rm -rf ~/.cache/huggingface/datasets/*needle_haystack*
rm -rf ~/.cache/huggingface/modules/*needle_haystack*

# Clear any arrow files
echo "2. Clearing arrow cache files..."
find ~/.cache/huggingface -name "*needle_haystack*.arrow" -delete 2>/dev/null

# Clear dataset info files
echo "3. Clearing dataset info files..."
find ~/.cache/huggingface -name "*needle_haystack*dataset_info.json" -delete 2>/dev/null

# Clear lock files
echo "4. Clearing lock files..."
find ~/.cache/huggingface -name "*needle_haystack*.lock" -delete 2>/dev/null

# Also check in the evaluation directory
if [ -d "evaluation/needle_haystack/__pycache__" ]; then
    echo "5. Clearing local pycache..."
    rm -rf evaluation/needle_haystack/__pycache__
fi

# Clear any .pyc files
echo "6. Clearing compiled Python files..."
find evaluation/needle_haystack -name "*.pyc" -delete 2>/dev/null

echo "Cache cleanup complete!"
echo ""
echo "Now run your evaluation again. The dataset will be regenerated with correct metadata."