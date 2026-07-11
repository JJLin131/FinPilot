# Task 1 实现报告

## RED

- 已先写入生命周期事件测试。
- 命令：`..\\..\\.venv\\Scripts\\python.exe -m pytest tests/test_context_budget.py -k "lifecycle_event or does_not_emit_events" -q`
- 结果：测试收集失败，`ImportError: cannot import name 'ContextLifecycleEvent'`。失败原因与任务要求一致：事件模型和回调接口尚未实现。

## GREEN

- 新增 `ContextEventKind`、`ContextLifecycleEvent` 和 `ContextEventCallback`。
- `ContextBuilder.build` 发布构建开始、压缩开始、压缩完成、构建完成及失败事件；回调异常只记录 debug 日志，不影响上下文构建。
- `build_global_prompt_bundle` 接收并透传 `event_callback`。
- 生命周期聚焦测试：`3 passed, 16 deselected`。
- 简报指定回归集：`24 passed`。
- 完整测试集：`173 passed, 2 warnings`。
- Ruff：目标文件 `All checks passed!`。
- `git diff --check`：通过；仅有 Windows 行尾转换提示。

## 关注点

- 全量测试中的两个警告来自虚拟环境依赖：FastAPI/Starlette TestClient 与 LangGraph serializer 的弃用提示，不是本次改动产生。
- 简报中的 `-k "lifecycle_event or does_not_emit_events"` 因测试名不含 `lifecycle_event`，实际只命中一个新测试；已额外用覆盖三个新行为的筛选命令验证为 `3 passed`。

## Task 1 审查修复

### RED

- 命令：`..\\..\\.venv\\Scripts\\python.exe -m pytest tests/test_context_budget.py -k "redacted_build_failed or global_prompt_bundle_forwards" -q`
- 输出：`1 failed, 1 passed, 19 deselected`。失败断言确认事件序列已产生，但 `build_failed.error` 仍为敏感异常文本 `prompt secret user text`。

### GREEN

- `build_failed` 改为固定错误码 `context_build_failed`，不再序列化异常文本。
- 新增冻结的 `_ContextBudget`，由单一 `_resolve_budget` 解析预算；构建开始和压缩事件使用该快照，最终 `build_finished` 字段由 `ContextUsage.lifecycle_event_fields()` 映射。
- 新增失败事件序列/脱敏测试及 `build_global_prompt_bundle` 回调透传测试。
- 定向命令：`..\\..\\.venv\\Scripts\\python.exe -m pytest tests/test_context_budget.py -k "redacted_build_failed or global_prompt_bundle_forwards or lifecycle_event or does_not_emit_events or callback_failure" -q`
- 输出：`4 passed, 17 deselected`。

### 修复验证

- 命令：`..\\..\\.venv\\Scripts\\python.exe -m pytest tests/test_context_budget.py tests/test_context_summarization.py -q`
- 输出：`26 passed in 0.41s`。
- 命令：`..\\..\\.venv\\Scripts\\python.exe -m ruff check finpilot\\context\\compression.py finpilot\\context\\builders.py tests\\test_context_budget.py`
- 输出：`All checks passed!`
- 命令：`git diff --check`
- 输出：通过；仅提示 Windows 工作副本的 LF/CRLF 行尾转换。

## Commit

- 前一阶段实现：`b83617a feat: 发布全局上下文生命周期事件`
- 本次审查修复：`686aaf4 修复上下文生命周期事件脱敏与预算映射`
