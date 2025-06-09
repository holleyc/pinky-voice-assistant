 #!/usr/bin/env bash
set -euo pipefail

# Path to your desired-models list
MODELS_FILE="models.txt"

# get installed models (first field of each line, skip header)
installed=($(ollama list | awk 'NR>1 {print $1}'))

# helper: check if an element is in an array
contains() {
  local e
  for e in "${installed[@]}"; do [[ "$e" == "$1" ]] && return 0; done
  return 1
}

while IFS= read -r model || [[ -n "$model" ]]; do
  # skip blank or commented lines
  [[ -z "$model" || "$model" =~ ^# ]] && continue

  if contains "$model"; then
    echo "✓ $model already installed"
  else
    echo "⭳ Pulling $model…"
    ollama pull "$model"
  fi
done < "$MODELS_FILE"
