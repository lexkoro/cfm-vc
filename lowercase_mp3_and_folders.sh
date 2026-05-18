#!/usr/bin/env bash
set -euo pipefail

# 1. Lowercase the .mp3 files
find "/home/alex/Data" -type f -iname "*.mp3" -print0 | while IFS= read -r -d '' file; do
    dir=$(dirname "$file")
    base=$(basename "$file")
    new_base="${base,,}"

    if [ "$base" != "$new_base" ]; then
        mv -v "$file" "$dir/$new_base"
    fi
done

# 2. Lowercase the folders (bottom-up)
find "/home/alex/Data" -mindepth 1 -depth -type d -print0 | while IFS= read -r -d '' folder; do
    parent=$(dirname "$folder")
    base=$(basename "$folder")
    new_base="${base,,}"

    if [ "$base" != "$new_base" ]; then
        mv -v "$folder" "$parent/$new_base"
    fi
done
