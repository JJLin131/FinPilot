# CLI Execution Plan Bordered Dialog Design

## Goal

将持久聊天界面中的纯文本执行计划改为单个带边框的终端对话框，使节点状态、耗时、Agent、任务与依赖关系更容易扫描，同时保持窄屏稳定。

## Layout

- 外层使用与 `You`、`FinPilot` 消息一致的 `╭─ / │ / ├─ / ╰─` 边框语言，标题为 `Execution Plan`。
- 每个 DAG 节点占一个分区；节点之间使用横向分隔线，不创建多个独立对话框。
- 分区第一行显示状态图标、状态和耗时；随后分别显示 `Node`、`Agent`、`Task`、`Depends`。
- `Task` 保留完整内容并允许自然换行；状态、耗时、Node 与 Agent 不与 Task 共用同一行。
- 动态执行区和最终写入聊天历史的计划摘要使用同一套布局。

## Runtime Behavior

- `PENDING`、`RUNNING`、`SUCCEEDED`、`FAILED`、`BLOCKED`、`SKIPPED` 使用不同图标与既有终端颜色。
- 运行事件只更新节点内容，不重建业务计划，也不修改 `AgentChatResponse`。
- 根据当前终端列宽生成上下边框和分隔线；最小宽度下字段值仍可换行，边框保持闭合。
- 保留现有 Planner、Agent 和 final-answer 动态状态，以及输出光标的稳定快照机制。

## Error Handling

- 缺少 Agent、Task 或 Depends 数据时显示空值或 `—`，不让渲染异常中断聊天。
- 超长 Node 或 Agent 在字段值区域自然换行；状态词与耗时保持完整。

## Tests

- 验证对话框包含完整上下边框、标题和节点分隔线。
- 验证三个节点在同一外框内分区显示。
- 验证长中文/英文 Task 不会拆开状态词。
- 验证动态计划与最终历史摘要使用一致结构。
- 保留 Prompt Toolkit 光标竞态回归测试，并运行 CLI 定向测试及完整测试集。
