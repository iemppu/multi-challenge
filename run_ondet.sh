#!/bin/bash
# Run all models with inject="on_detection" configuration

timestamp=$(date +%Y%m%d%H%M%S)
mkdir -p logs

echo "=== Running on_detection mode ==="

# 运行所有 Table 2/3 模型 (model23)
for cfg in scripts/model23/ondet/*.yaml; do
    echo "Running: $cfg"
    python scripts/interface_openrouter.py --config "$cfg" run |& tee "logs/${timestamp}-$(basename $cfg .yaml)-ondet.log"
done

# 运行所有 Table 5 模型 (model5)
for cfg in scripts/model5/ondet/*.yaml; do
    echo "Running: $cfg"
    python scripts/interface_openrouter.py --config "$cfg" run |& tee "logs/${timestamp}-$(basename $cfg .yaml)-ondet.log"
done

echo "=== Done ==="
