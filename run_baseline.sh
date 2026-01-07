#!/bin/bash
# Run all models with disable_controller=true (no SLSM baseline)

timestamp=$(date +%Y%m%d%H%M%S)
mkdir -p logs

echo "=== Running baseline mode (no SLSM) ==="

# 运行所有 Table 2/3 模型 (model23)
for cfg in scripts/model23/baseline/*.yaml; do
    echo "Running: $cfg"
    python scripts/interface_openrouter.py --config "$cfg" run |& tee "logs/${timestamp}-$(basename $cfg .yaml)-baseline.log"
done

# 运行所有 Table 5 模型 (model5)
for cfg in scripts/model5/baseline/*.yaml; do
    echo "Running: $cfg"
    python scripts/interface_openrouter.py --config "$cfg" run |& tee "logs/${timestamp}-$(basename $cfg .yaml)-baseline.log"
done

echo "=== Done ==="
