#!/bin/bash
# Evaluate all always mode results in data/final_model_responses/

timestamp=$(date +%Y%m%d%H%M%S)
mkdir -p logs outputs

echo "=== Evaluating always mode results ==="

for resp in data/final_model_responses/*-always.jsonl; do
    if [ -f "$resp" ]; then
        name=$(basename "$resp" .jsonl)
        echo ""
        echo "Evaluating: $name"
        python scripts/interface_openrouter.py eval --responses "$resp" |& tee "logs/${timestamp}-eval-${name}.log"
    fi
done

echo ""
echo "=== Done ==="
