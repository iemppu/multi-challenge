
---

## 2025-01-05 04:45

**Q:** prompt发我看看 然后解释一下verify clarify是什么

**A:**

### Controller Prompt (核心部分)

```
Your job:
1) Extract stable user facts/preferences as facts.
2) Track constraints explicitly stated in the conversation.
3) Track edits/revisions: if user changes a requirement, add an edit.
4) Mark contradictions as assumption.status="contradicted" or constraint.status="violated".
5) Set plan.mode:
   - "verify" if any contradiction/violation exists that can be resolved using existing context.
   - "clarify" if a task-critical unknown remains open.
   - "proceed" otherwise.
6) required_fixes must be concrete, action-oriented instructions for the answering model.
```

### verify 和 clarify 的区别

| 模式 | 触发条件 | 含义 | 举例 |
|------|---------|------|------|
| **proceed** | 无问题 | 正常回答 | 一切正常 |
| **verify** | 检测到矛盾/违反 | 需要验证/修正，但**信息已足够** | 用户说"不要emoji"，但之前回答用了emoji |
| **clarify** | 关键信息缺失 | 需要澄清，但**不能问用户**，只能添加 required_fixes | 用户要求"写个函数"但没说用什么语言 |

### verify 示例
```
用户: 帮我写邮件，不要太正式
助手: Dear Sir/Madam, I am writing to... (太正式了)
用户: 重写一下

Controller 检测: constraint "不要太正式" 被违反
plan.mode = "verify"
required_fixes = ["使用更口语化的语气"]
```

### clarify 示例
```
用户: 帮我写个排序函数

Controller 检测: 缺少关键信息（什么语言？升序还是降序？）
plan.mode = "clarify"
required_fixes = ["假设使用Python", "假设升序排列"]
```

### 何时触发注入

```python
risk_modes = ["verify", "clarify"]

if inject == "on_risk" and plan.mode in risk_modes:
    inject_note = True  # 注入 Memory Note
```

只有当 `plan.mode` 是 `verify` 或 `clarify` 时，`on_risk` 模式才会注入 Memory Note。
