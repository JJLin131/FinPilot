# FinPilot CLI 全局上下文状态设计

## 目标

将 `finpilot chat` 改为持久化终端应用，使输入框和底栏在等待输入、模型思考、上下文压缩与回答渲染期间始终可见。底栏只展示当前会话的 `global_default` 全局上下文状态，不重复模型配置，也不显示地址或数据来源字段。

## 数据口径

- 进度条只使用 `context_usage["global"]`，对应 `global_default` 策略。
- 当前占用使用该运行态快照的 `estimated_tokens_after`。
- 分母使用同一快照的 `effective_input_budget`，确保预留输出 token 后的可用输入预算与进度条一致。
- 百分比为 `estimated_tokens_after / effective_input_budget`，显示值限制在 `0%` 到 `100%`；超预算状态仍显示明确的 `over budget` 文案。
- 当前 token 计数器是启发式估算，因此数值前显示 `~`，不宣称为模型供应商返回的精确 usage。
- 本轮尚未产生运行态快照时，显示 `context waiting` 和空进度条；完成一轮后保留该轮最后一个 global 快照，直到下一轮开始更新。
- 所有预算和阈值从运行态 `ContextUsage` 读取，不在 CLI 中硬编码 `384k` 或其他模型窗口值。

## 终端布局

CLI 使用一个长期存活的 `prompt_toolkit.Application`，布局由三部分组成：

1. 可滚动的对话输出区，承载用户消息、FinPilot 回答、证据和必要的系统通知。
2. 黄色完整线框输入区，提交后暂时禁用编辑，但保持可见。
3. 输入框下方两行状态区。

状态区第一行显示 `global context`、连续进度条、`~已用 / 有效预算`、百分比和状态；第二行只显示 `user`、`chat`、`debug`。不显示 `source`、模型配置或服务地址。

颜色语义保持稳定：青色表示正常可用，黄色表示接近触发阈值或正在压缩，红色表示超预算或压缩失败，绿色用于用户身份和成功状态。有效内容不使用暗色，避免被理解为未生效。

## 状态与动画

全局上下文状态机包含：

- `waiting`：当前会话还没有运行态 global 快照。
- `ready`：占用低于压缩触发阈值。
- `near limit`：占用达到 `trigger_tokens`，但当前没有执行压缩。
- `compressing`：global 压缩开始，黄色进度条做低频呼吸动画，并显示持续时间。
- `compressed`：本轮发生压缩且成功，短暂显示压缩前后数值，再稳定为 `ready`。
- `over budget`：压缩后仍超过有效输入预算，使用红色状态。
- `failed`：压缩过程异常，显示简短错误状态，详细诊断仅在 debug 输出中出现。

现有 `FinPilot is thinking` 动画继续存在，并显示已思考时间。进入 global 压缩时，思考文案切换为 `FinPilot is compressing context`；压缩结束后回到当前处理阶段。压缩提示只在固定状态区和思考区更新，不插入聊天记录。

## 运行时数据流

`ContextBuilder` 增加可选的上下文事件回调，发布 global 构建和压缩生命周期事件。事件包含阶段、策略、估算 token、有效预算、触发阈值、压缩状态和耗时，不包含完整 prompt 内容。

回调沿 `FinPilotGraph -> FinPilotService -> CLI chat session` 注入。非 CLI 调用不传回调时保持现有同步行为和公开 `chat(user_id, chat_id, content)` 兼容性。

持久化 CLI 在后台工作线程调用服务，事件通过线程安全队列发送给 UI 线程。UI 线程消费事件、更新状态模型并调用 `Application.invalidate()`，避免后台线程直接修改终端组件。

每轮最终响应中的 `route_debug.context_usage.global` 作为权威的轮次完成快照，用于校正事件流中的临时值，并供 `/status` 展示。CLI 在当前进程内按 `user_id + chat_id` 保存最近一次快照；`/new` 切换到空状态，当前进程内的 `/resume` 恢复对应快照。现有聊天存储不持久化 `context_usage`，因此 CLI 重启后的 `/resume` 明确显示 `waiting`，在恢复会话完成下一轮请求后刷新，不为本功能扩张数据库结构。

## `/status` 行为

`/context` 继续作为 `/status` 的兼容别名。`/status` 的 Context 区优先展示当前会话最近一次运行态 `global_default` 快照，包括：

- 策略名与阶段；
- 压缩前、压缩后和有效输入预算；
- 触发阈值；
- 是否发生压缩、是否在预算内；
- token 计数器类型；
- 最近一次 global 压缩事件摘要。

没有运行态快照时才回退到历史消息启发式估算，并明确标记为 `history estimate`，不再与真实运行态快照混合展示。

## 错误处理

- 后台服务调用失败时恢复输入能力，在对话区显示错误通知，底栏状态回到最近一次稳定快照。
- 上下文事件缺字段或顺序异常时忽略无效事件，并使用最终响应快照校正。
- 终端宽度不足时，第一行进度条缩短，数值和状态移到下一视觉行；会话信息可换行但不得被截断。
- 不支持 ANSI 动画的终端显示静态颜色和持续时间，功能不依赖动画成立。

## 测试范围

- `ContextBuilder` 生命周期事件：未压缩、压缩成功、压缩后超预算和异常路径。
- 服务与图回调透传：未注入回调时保持兼容，注入时只发布 global 事件。
- CLI 状态模型：各状态转换、耗时、百分比、颜色阈值和最终快照校正。
- 持久化应用：提交期间输入禁用、底栏常驻、后台完成后恢复输入。
- Slash 命令：`/status`、`/context`、`/new`、`/resume` 与不同 chat 快照隔离。
- 窄终端布局：状态换行且不丢失 `user/chat/debug`。
- 回归测试：`FinPilot is thinking`、审批提示暂停与恢复、会话级审批复用保持有效。

## 非目标

- 不引入真实 token streaming。
- 不把局部 Agent、planner 或 answer 阶段预算放入底栏。
- 不从外部模型 API 查询或硬编码上下文窗口。
- 不修改模型配置、RAG 行为、上下文压缩策略本身或业务处理结果。
