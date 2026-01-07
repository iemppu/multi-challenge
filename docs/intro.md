# SLSM Multi-Challenge 配置指南

## 4种 inject 模式的处理流程

```
原始对话: [user, assistant, user, assistant, ...]
```

### 1. `baseline` (disable_controller=true)
```
原始对话 ──────────────────────────────────> 底层LLM ──> 响应
         (不调用 controller，不消耗额外 token)
```
**成本**: 1次底层LLM调用

---

### 2. `onrisk` (inject="on_risk") - 默认
```
原始对话 ──> Controller 逐轮分析 ──> SemanticState
                                        │
                                        ▼
                              plan.mode == "verify"/"clarify"?
                                   │            │
                                  YES          NO
                                   │            │
                                   ▼            ▼
                        原始对话 + [MEMORY NOTE]   原始对话
                                   │            │
                                   └─────┬──────┘
                                         ▼
                                     底层LLM ──> 响应
```
**成本**: N次Controller调用 + 1次底层LLM调用

---

### 3. `always` (inject="always")
```
原始对话 ──> Controller 逐轮分析 ──> SemanticState
                                        │
                                        ▼
                        原始对话 + [MEMORY NOTE] (总是注入)
                                        │
                                        ▼
                                    底层LLM ──> 响应
```
**成本**: N次Controller调用 + 1次底层LLM调用

---

### 4. `ondet` (inject="on_detection") - 双轮制
```
原始对话 ──> Controller 逐轮分析 ──> SemanticState
                                        │
                                        ▼
                    ┌─────── 第1轮：原始对话 (不注入) ───────┐
                    │                                       │
                    ▼                                       │
                底层LLM ──> y0 (候选答案)                    │
                    │                                       │
                    ▼                                       │
        Controller 检测 y0 是否违反约束                      │
                    │                                       │
           有冲突?  │                                       │
            │      │                                        │
           YES    NO ──────────────────────────────> 返回 y0
            │
            ▼
    第2轮：原始对话 + [MEMORY NOTE] (强制注入)
            │
            ▼
        底层LLM ──> y1 ──> 返回 y1
```
**成本**: N次Controller调用 + 1~2次底层LLM调用 + 1次冲突检测

---

## 模式对比总结

| 模式 | Controller调用 | 底层LLM调用 | 注入条件 | 适用场景 |
|------|---------------|------------|---------|---------|
| `baseline` | 0 | 1 | 无 | 纯baseline对比 |
| `onrisk` | N | 1 | 检测到风险时 | 平衡成本与效果 |
| `always` | N | 1 | 总是 | 最大化约束遵守 |
| `ondet` | N+1 | 1~2 | 检测到冲突时 | 避免不必要干预 |

---

## Memory Mode: flat vs structured

### Structured (默认)

状态格式:
```json
SemanticState {
  "facts": [{"id": "F1", "text": "用户喜欢简洁", "evidence": "..."}],
  "constraints": [{"id": "C1", "text": "不要用emoji", "status": "satisfied"}],
  "assumptions": [{"id": "A1", "text": "...", "status": "valid"}],
  "unknowns": [...],
  "edits": [...],
  "plan": {"mode": "proceed", "reasons": [...]}
}
```

注入的 Memory Note:
```
[SLSM MEMORY NOTE]
Constraints:
- [satisfied] 不要用emoji
User facts/preferences:
- 用户喜欢简洁
```

---

### Flat (memory_mode="flat")

状态格式:
```json
FlatState {
  "memory_text": "用户喜欢简洁风格，不要emoji，之前要求过修改标题...",
  "plan": {"mode": "proceed"},
  "mismatch": false
}
```

注入的 Memory Note:
```
[SLSM MEMORY NOTE]
用户喜欢简洁风格，不要emoji，之前要求过修改标题...
```

---

### 对比

| 特性 | Structured | Flat |
|------|-----------|------|
| 状态格式 | 结构化JSON (facts/constraints/...) | 自由文本 |
| 解析复杂度 | 高 (需要精确JSON) | 低 (自然语言) |
| 信息组织 | 分类明确 | 混合在一起 |
| 冲突检测 | 可精确检测 (status字段) | 依赖语义理解 |
| Token消耗 | 较多 (JSON结构) | 较少 |
| 适用场景 | 需要精确约束追踪 | 简单记忆场景 |

---

## 配置示例

### YAML 配置
```yaml
slsm:
  disable_controller: false     # true = baseline模式
  memory_mode: "structured"     # "structured" 或 "flat"
  inject: "on_risk"             # "never" | "on_risk" | "always" | "on_detection"
  risk_modes: ["verify", "clarify"]
  note_max_items: 6
  gate_facts_by_evidence: true
```

### 目录结构
```
scripts/
├── model5/
│   ├── always/      # inject: "always"
│   ├── baseline/    # disable_controller: true
│   ├── never/       # inject: "never" (不推荐)
│   ├── ondet/       # inject: "on_detection"
│   └── onrisk/      # inject: "on_risk" (默认)
└── model23/
    ├── always/
    ├── baseline/
    ├── ondet/
    └── ...
```

### 运行脚本
```bash
./run_always.sh    # 运行 always 模式
./run_ondet.sh     # 运行 on_detection 模式
./run_baseline.sh  # 运行 baseline 模式 (无SLSM)
```

---

## 注意事项

1. **`inject: "never"`**: Controller 仍然运行但不注入，浪费资源，不推荐使用
2. **`disable_controller: true`**: 完全跳过 Controller，是真正的 baseline
3. **`on_detection`**: 成本最高（可能2次LLM调用），但干预最精准
4. **`always`**: 最激进，每次都注入 Memory Note
