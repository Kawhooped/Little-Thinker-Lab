#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/import_full_package.sh /path/to/littlethinker_laughing_overflow
#
# The source directory must contain:
#   littlethinker_overflow.pt
#   training_data.jsonl

SOURCE_DIR="${1:-}"
if [[ -z "$SOURCE_DIR" || ! -d "$SOURCE_DIR" ]]; then
  echo "Usage: $0 /path/to/littlethinker_laughing_overflow" >&2
  exit 2
fi

for file in littlethinker_overflow.pt training_data.jsonl; do
  if [[ ! -f "$SOURCE_DIR/$file" ]]; then
    echo "Missing required file: $SOURCE_DIR/$file" >&2
    exit 2
  fi
done

mkdir -p models/laughing-overflow data/mosslight
cp "$SOURCE_DIR/littlethinker_overflow.pt" models/laughing-overflow/
cp "$SOURCE_DIR/training_data.jsonl" data/mosslight/

git add models/laughing-overflow/littlethinker_overflow.pt data/mosslight/training_data.jsonl
git commit -m "Add trained model weights and Mosslight training corpus"

echo "Full model assets added. Push with: git push origin HEAD"
