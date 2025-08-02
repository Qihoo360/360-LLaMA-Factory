#!/bin/bash

# Script to clear needle haystack dataset cache

echo "Clearing needle haystack dataset cache..."

# Clear HuggingFace datasets cache for needle_haystack
if [ -d ~/.cache/huggingface/datasets ]; then
    echo "Removing cached needle_haystack datasets..."
    rm -rf ~/.cache/huggingface/datasets/*needle_haystack*
    echo "Cache cleared!"
else
    echo "No HuggingFace cache directory found."
fi

# Alternative cache locations
if [ -d ~/.cache/huggingface/modules ]; then
    echo "Clearing modules cache..."
    rm -rf ~/.cache/huggingface/modules/*needle_haystack*
fi

# Clear any local cache
if [ -d ./cache ]; then
    echo "Clearing local cache..."
    rm -rf ./cache/*needle_haystack*
fi

echo "Done! The next evaluation will use your updated configuration."