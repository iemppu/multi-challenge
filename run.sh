python scripts/interface_openrouter.py  --config scripts/model23/claude-3.5-sonnet.yaml run|& tee logs/claude-3.5-sonnet.log
timestamp=$(date +%Y%m%d%H%M%S)

  # 运行所有 Table 2/3 模型
  for cfg in scripts/model23/*.yaml; do
      echo "Running: $cfg"
      python scripts/interface_openrouter.py  --config "$cfg" run|& tee logs/$timestamp-$cfg.log
  done

  # 运行所有 Table 5 模型
  for cfg in scripts/model5/*.yaml; do
      echo "Running: $cfg"
      python scripts/interface_openrouter.py  --config "$cfg" run|& tee logs/$timestamp-$cfg.log
  done
