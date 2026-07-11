# FinPilot CLI Global Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `finpilot chat` 改为持久化终端应用，在输入框下方实时展示 `global_default` 全局上下文占用和压缩状态。

**Architecture:** `ContextBuilder` 通过可选回调发布不含 prompt 正文的上下文生命周期事件，`FinPilotGraph` 和 `FinPilotService` 只负责依赖注入。新的 `finpilot.cli_chat` 模块在后台线程调用同步服务，通过线程安全队列把事件交回 `prompt_toolkit` UI 线程，并用最终响应中的 `route_debug.context_usage.global` 校正底栏状态。

**Tech Stack:** Python 3.11、Pydantic、LangGraph、Typer、Rich、prompt_toolkit、pytest、ruff

## Global Constraints

- 底栏只显示 `global_default`，不显示局部 Agent、planner、answer、模型配置、服务地址或 `source`。
- 当前占用使用 `estimated_tokens_after`，分母使用 `effective_input_budget`，数值前显示 `~`。
- 所有预算与阈值从运行态 `ContextUsage` 读取，禁止在 CLI 中硬编码 `384k`。
- 输入框与底栏在 waiting、thinking、compressing 和 answer rendering 期间保持可见。
- 保留 `FinPilot is thinking`、思考耗时、审批提示暂停与会话级审批复用。
- 不修改压缩策略、RAG 行为、模型配置、数据库结构或业务回答。
- 现有 `.env.example` 工作区改动不属于本计划，任何提交均不得暂存该文件。

---

### Task 1: 发布全局上下文生命周期事件

**Files:**
- Modify: `finpilot/context/compression.py:72-97,271-380`
- Modify: `finpilot/context/builders.py:116-129`
- Test: `tests/test_context_budget.py`

**Interfaces:**
- Produces: `ContextLifecycleEvent`, `ContextEventCallback`, `ContextBuilder.build(..., event_callback=None)`
- Produces: `build_global_prompt_bundle(..., event_callback=None)`
- Consumes: existing `ContextUsage` and `PromptContextBundle`

- [ ] **Step 1: Write failing lifecycle-event tests**

```python
def test_global_context_builder_emits_build_and_compression_events():
    events: list[ContextLifecycleEvent] = []
    builder = ContextBuilder(
        policies={
            "global_default": ContextPolicy(
                token_budget=160,
                trigger_ratio=0.25,
                reserved_output_tokens=10,
                compression_rules=[
                    CompressionRule(
                        name="recent_messages",
                        path="session.recent_messages",
                        method="deterministic",
                        target_tokens=40,
                        priority=10,
                        preserve_last=1,
                    )
                ],
            )
        },
        summarizer=FakeSummarizer(),
    )

    bundle = builder.build(
        "global_default",
        [ContextSegment(name="session", value={"recent_messages": [{"content": "x" * 2000}]})],
        stage="global",
        event_callback=events.append,
    )

    assert [event.kind for event in events] == ["build_started", "compression_started", "compression_finished", "build_finished"]
    assert all(event.stage == "global" for event in events)
    assert all(event.policy == "global_default" for event in events)
    assert events[-1].estimated_tokens == bundle.usage.estimated_tokens_after
    assert events[-1].effective_input_budget == bundle.usage.effective_input_budget
    assert events[-1].within_budget == bundle.usage.within_budget


def test_context_builder_does_not_emit_events_without_callback():
    builder = ContextBuilder(policies={"plain": ContextPolicy(token_budget=1000)}, summarizer=FakeSummarizer())
    bundle = builder.build("plain", [ContextSegment(name="session", value={"user_message": "hello"})], stage="global")
    assert bundle.status == "ready"


def test_context_callback_failure_does_not_break_context_build():
    def broken_callback(event: ContextLifecycleEvent) -> None:
        raise RuntimeError("ui closed")

    builder = ContextBuilder(policies={"plain": ContextPolicy(token_budget=1000)}, summarizer=FakeSummarizer())
    bundle = builder.build(
        "plain",
        [ContextSegment(name="session", value={"user_message": "hello"})],
        stage="global",
        event_callback=broken_callback,
    )
    assert bundle.status == "ready"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_context_budget.py -k "lifecycle_event or does_not_emit_events" -q`

Expected: FAIL because `ContextLifecycleEvent` and `event_callback` do not exist.

- [ ] **Step 3: Add the event model and callback type**

```python
ContextEventKind = Literal[
    "build_started",
    "compression_started",
    "compression_finished",
    "build_finished",
    "build_failed",
]


class ContextLifecycleEvent(BaseModel):
    kind: ContextEventKind
    stage: Literal["global", "planner", "decision", "answer"]
    policy: str
    estimated_tokens: int
    effective_input_budget: int
    trigger_tokens: int
    compressed: bool = False
    within_budget: bool | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


ContextEventCallback = Callable[[ContextLifecycleEvent], None]
```

- [ ] **Step 4: Emit events from `ContextBuilder.build`**

Add `event_callback: ContextEventCallback | None = None` to `build`. Use a local monotonic start time and a helper that never lets a UI callback break context construction:

```python
def emit(kind: ContextEventKind, **updates: Any) -> None:
    if event_callback is None:
        return
    event = ContextLifecycleEvent(
        kind=kind,
        stage=stage,
        policy=policy_name,
        estimated_tokens=int(updates.pop("estimated_tokens", payload_tokens_before)),
        effective_input_budget=effective_input_budget,
        trigger_tokens=trigger_tokens,
        elapsed_seconds=time.perf_counter() - started_at,
        **updates,
    )
    try:
        event_callback(event)
    except Exception:
        logger.debug("Context lifecycle callback failed", exc_info=True)
```

Emit `build_started` after budgets are resolved, `compression_started` immediately before applying compression rules, `compression_finished` after rules complete, and `build_finished` immediately before returning the bundle. Wrap the build body so an exception emits `build_failed` and is re-raised. Do not include payloads, summaries, user text, or tool outputs in events.

- [ ] **Step 5: Thread the callback through the global builder**

```python
def build_global_prompt_bundle(
    state: GraphState,
    *,
    summary_cache: dict[str, Any] | None = None,
    event_callback: ContextEventCallback | None = None,
) -> PromptContextBundle:
    session_context = build_session_context(state)
    return _context_builder.build(
        "global_default",
        [ContextSegment(name="session", value=session_context, priority=10)],
        stage="global",
        query=state.user_message,
        summary_cache=summary_cache,
        event_callback=event_callback,
    )
```

- [ ] **Step 6: Run focused and regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_context_budget.py tests/test_context_summarization.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add finpilot/context/compression.py finpilot/context/builders.py tests/test_context_budget.py
git commit -m "feat: 增加全局上下文生命周期事件"
```

### Task 2: 贯通 Graph 与 Service 的事件回调

**Files:**
- Modify: `finpilot/agent/graph.py:39-60,165-204,320-325`
- Modify: `finpilot/agent/service.py:18-38`
- Test: `tests/test_safety_integration.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ContextEventCallback` from Task 1
- Produces: `FinPilotGraph(..., context_event_callback=None)`
- Produces: `FinPilotService(..., context_event_callback=None)`
- Preserves: `FinPilotService.chat(user_id, chat_id, content)`

- [ ] **Step 1: Write failing graph callback test**

```python
def test_graph_forwards_only_global_context_lifecycle_events(monkeypatch):
    events: list[ContextLifecycleEvent] = []
    graph = build_test_graph(context_event_callback=events.append)
    response = graph.run("user-1", "chat-1", "rule")

    assert response.status == "SUCCEEDED"
    assert events
    assert {event.stage for event in events} == {"global"}
    final_global = response.route_debug["context_usage"]["global"]
    assert events[-1].estimated_tokens == final_global["estimated_tokens_after"]
```

- [ ] **Step 2: Run test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_safety_integration.py -k "forwards_only_global" -q`

Expected: FAIL because `FinPilotGraph` does not accept the callback.

- [ ] **Step 3: Inject the callback into graph global builds**

```python
class FinPilotGraph:
    def __init__(
        self,
        router: IntentRouter,
        tools: ToolRegistry,
        audit_store: AuditStore,
        memory_manager: MemoryManager,
        safety: SafetyReviewService | None = None,
        planner: ExecutionPlanningService | None = None,
        answering_service: FinanceAnsweringService | None = None,
        context_event_callback: ContextEventCallback | None = None,
    ):
        self.context_event_callback = context_event_callback
```

Pass `event_callback=self.context_event_callback` to both `build_global_prompt_bundle` calls in `_context_load` and post-execution global refresh. Do not pass it to planner, decision, or answer bundle builders.

- [ ] **Step 4: Add service injection without changing `chat`**

```python
class FinPilotService:
    def __init__(
        self,
        audit_store: AuditStore | None = None,
        memory_manager: MemoryManager | None = None,
        safety: SafetyReviewService | None = None,
        context_event_callback: ContextEventCallback | None = None,
    ):
        ...
        self.graph = FinPilotGraph(
            self.router,
            self.tools,
            self.audit_store,
            self.memory_manager,
            safety=self.safety,
            context_event_callback=context_event_callback,
        )
```

- [ ] **Step 5: Add compatibility test for existing factories**

```python
def test_default_service_factory_accepts_context_event_callback(monkeypatch):
    captured: dict[str, object] = {}

    class RecordingService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli, "FinPilotService", RecordingService)
    callback = lambda event: None
    cli._default_service_factory(context_event_callback=callback)
    assert captured["context_event_callback"] is callback
```

- [ ] **Step 6: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_safety_integration.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add finpilot/agent/graph.py finpilot/agent/service.py tests/test_safety_integration.py tests/test_cli.py
git commit -m "feat: 贯通全局上下文运行态回调"
```

### Task 3: 建立 CLI 全局上下文状态模型

**Files:**
- Create: `finpilot/cli_chat.py`
- Modify: `finpilot/cli.py:602-689,1019-1048`
- Test: `tests/test_cli_chat.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ContextLifecycleEvent`, response `route_debug.context_usage.global`
- Produces: `GlobalContextViewState`, `GlobalContextSnapshotCache`, `format_global_context_fragments`
- Produces: `global_context_snapshot(response) -> dict[str, Any] | None`

- [ ] **Step 1: Write failing state-transition tests**

```python
def test_global_context_state_tracks_compression_and_final_snapshot():
    state = GlobalContextViewState()
    state.apply_event(
        ContextLifecycleEvent(
            kind="compression_started",
            stage="global",
            policy="global_default",
            estimated_tokens=320_000,
            effective_input_budget=380_000,
            trigger_tokens=323_000,
        )
    )
    assert state.status == "compressing"
    assert state.percent == pytest.approx(320_000 / 380_000)

    state.apply_snapshot(
        {
            "effective_policy": "global_default",
            "stage": "global",
            "estimated_tokens_before": 320_000,
            "estimated_tokens_after": 140_000,
            "effective_input_budget": 380_000,
            "trigger_tokens": 323_000,
            "compressed": True,
            "within_budget": True,
            "token_counter": "heuristic",
        }
    )
    assert state.status == "compressed"
    assert state.used_tokens == 140_000
    assert state.display_usage == "~140k / 380k"
```

```python
def test_context_cache_isolated_by_user_and_chat():
    cache = GlobalContextSnapshotCache()
    cache.put("user-1", "chat-a", {"estimated_tokens_after": 100})
    assert cache.get("user-1", "chat-a") == {"estimated_tokens_after": 100}
    assert cache.get("user-1", "chat-b") is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py -q`

Expected: FAIL because `finpilot.cli_chat` does not exist.

- [ ] **Step 3: Implement the state model and cache**

```python
@dataclass
class GlobalContextViewState:
    policy: str = "global_default"
    status: str = "waiting"
    used_tokens: int = 0
    effective_input_budget: int = 0
    trigger_tokens: int = 0
    before_tokens: int = 0
    compressed: bool = False
    within_budget: bool = True
    token_counter: str = "heuristic"
    status_started_at: float = field(default_factory=time.perf_counter)

    @property
    def percent(self) -> float:
        if self.effective_input_budget <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_tokens / self.effective_input_budget))

    @property
    def display_usage(self) -> str:
        if self.effective_input_budget <= 0:
            return "waiting"
        return f"~{format_tokens(self.used_tokens)} / {format_tokens(self.effective_input_budget)}"

    def apply_event(self, event: ContextLifecycleEvent) -> None:
        if event.stage != "global" or event.policy != "global_default":
            return
        self.used_tokens = event.estimated_tokens
        self.effective_input_budget = event.effective_input_budget
        self.trigger_tokens = event.trigger_tokens
        self.status = {
            "compression_started": "compressing",
            "build_failed": "failed",
        }.get(event.kind, self.status)
        if event.kind in {"compression_started", "build_failed"}:
            self.status_started_at = time.perf_counter()

    def apply_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.used_tokens = int(snapshot["estimated_tokens_after"])
        self.before_tokens = int(snapshot["estimated_tokens_before"])
        self.effective_input_budget = int(snapshot["effective_input_budget"])
        self.trigger_tokens = int(snapshot["trigger_tokens"])
        self.compressed = bool(snapshot["compressed"])
        self.within_budget = bool(snapshot["within_budget"])
        self.token_counter = str(snapshot.get("token_counter", "heuristic"))
        self.status = "over budget" if not self.within_budget else "compressed" if self.compressed else "near limit" if self.used_tokens >= self.trigger_tokens else "ready"
```

`GlobalContextSnapshotCache` is a small locked dictionary keyed by `(user_id, chat_id)`. It stores copied snapshot dictionaries and has `get`, `put`, and `clear_chat` methods.

- [ ] **Step 4: Implement fragment formatting with stable color semantics**

```python
def format_global_context_fragments(state: GlobalContextViewState, width: int) -> list[tuple[str, str]]:
    color = "ansired" if state.status in {"failed", "over budget"} else "ansiyellow" if state.status in {"near limit", "compressing"} else "ansicyan"
    cells = max(8, min(28, width // 5))
    filled = round(cells * state.percent)
    bar = "█" * filled + "░" * (cells - filled)
    elapsed = time.perf_counter() - state.status_started_at
    status = f"compressing · {elapsed:.1f}s" if state.status == "compressing" else state.status
    return [
        ("class:context.label", "global context "),
        (f"class:context.{color}", bar),
        ("class:context.value", f"  {state.display_usage}  {state.percent:.0%}  "),
        (f"class:context.{color}", status),
    ]
```

Use high-contrast styles for all active values; do not use `dim` for configured or effective state.

- [ ] **Step 5: Make `/status` prefer runtime snapshots**

Add an optional snapshot argument to `_build_status_snapshot` and `_render_status`. When present, render `estimated_tokens_before`, `estimated_tokens_after`, `effective_input_budget`, `trigger_tokens`, `compressed`, `within_budget`, and `token_counter`. When absent, retain history estimation but label it `history estimate`.

```python
def global_context_snapshot(response: AgentChatResponse) -> dict[str, Any] | None:
    debug = response.route_debug or {}
    usage = debug.get("context_usage")
    if not isinstance(usage, dict):
        return None
    snapshot = usage.get("global")
    return dict(snapshot) if isinstance(snapshot, dict) else None
```

- [ ] **Step 6: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add finpilot/cli_chat.py finpilot/cli.py tests/test_cli_chat.py tests/test_cli.py
git commit -m "feat: 增加 CLI 全局上下文状态模型"
```

### Task 4: 构建持久化 prompt_toolkit 对话应用

**Files:**
- Modify: `finpilot/cli_chat.py`
- Modify: `finpilot/cli.py:147-205,408-430,947-1016`
- Test: `tests/test_cli_chat.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: service factory compatible with `context_event_callback=` and `ApprovalService`
- Produces: `FinPilotChatApplication.run() -> None`
- Produces: `FinPilotChatApplication.submit(text: str) -> None`
- Produces: `render_response_fragments(response, debug) -> StyleAndTextTuples`
- Preserves: slash commands and `_run_chat` for `ask` and noninteractive tests

- [ ] **Step 1: Write failing persistent-UI tests**

```python
def test_submit_keeps_input_and_status_visible_while_worker_runs():
    gate = threading.Event()
    service = BlockingFakeService(gate=gate, response=fake_response_with_global_usage())
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        history=InMemoryHistory(),
        service_factory=lambda **kwargs: service,
        approval_service=ApprovalService(),
    )

    app.submit("hello")
    assert app.busy is True
    assert app.input_area.read_only is True
    assert app.input_area in list(app.layout.find_all_controls())
    assert app.context_state.status in {"waiting", "compressing"}

    gate.set()
    app.wait_for_worker(timeout=1)
    assert app.busy is False
    assert app.input_area.read_only is False
    assert app.context_state.status == "ready"
```

```python
def test_compression_event_updates_thinking_copy_and_elapsed_status():
    app = build_chat_application()
    app.handle_context_event(compression_started_event())
    assert app.context_state.status == "compressing"
    assert "FinPilot is compressing context" in app.thinking_text()


def test_response_renderer_preserves_answer_evidence_and_tool_calls():
    text = fragment_text(render_response_fragments(fake_response(), debug=True))
    assert "工资发放需要审批" in text
    assert "doc-1" in text
    assert "search_finance_knowledge" in text
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py -k "persistent or compression_event or submit_keeps" -q`

Expected: FAIL because `FinPilotChatApplication` is not implemented.

- [ ] **Step 3: Build the persistent layout**

Create one `Application(full_screen=False)` with:

```python
self.output_control = FormattedTextControl(self.output_fragments, focusable=False)
self.output_window = Window(self.output_control, wrap_lines=True, always_hide_cursor=True)
self.input_area = TextArea(
    multiline=False,
    history=history,
    prompt=HTML("<input.user>You</input.user> <input.symbol>›</input.symbol> "),
    accept_handler=self._accept_input,
)
self.context_control = FormattedTextControl(self.context_fragments)
self.session_control = FormattedTextControl(self.session_fragments)
self.container = HSplit([
    self.output_window,
    Frame(self.input_area, title=HTML("<input.title>Input</input.title>")),
    Window(self.context_control, height=1),
    Window(self.session_control, height=1),
])
```

Use `DynamicContainer` or `ConditionalContainer` only where terminal width requires wrapping. Keep the input frame and both status rows present for the application lifetime.

Render existing Rich panels and tables into ANSI text with a dedicated recording `Console`, then convert that ANSI output using `prompt_toolkit.formatted_text.ANSI`. This keeps the existing answer, issues, evidence, tool calls and debug details inside the persistent scrollable output instead of printing outside the application. The renderer must not write to the process-global `console`.

- [ ] **Step 4: Execute chat work in one background thread**

```python
def submit(self, text: str) -> None:
    if self.busy or not text.strip():
        return
    self.busy = True
    self.input_area.read_only = True
    self._append_user_message(text)
    self._thinking_started_at = time.perf_counter()
    self._worker = threading.Thread(target=self._run_request, args=(text,), daemon=True)
    self._worker.start()
    self.application.invalidate()


def _run_request(self, text: str) -> None:
    try:
        service = self.service_factory(
            interactive_approval=True,
            approval_service=self.approval_service,
            context_event_callback=self._event_queue.put,
        )
        response = service.chat(self.user_id, self.chat_id, text)
        self._event_queue.put(("response", response))
    except Exception as exc:
        self._event_queue.put(("error", str(exc)))
    finally:
        self._event_queue.put(("complete", None))
```

Schedule a UI-thread drain with `application.loop.call_soon_threadsafe(self._drain_events)` after each queue write. Only `_drain_events` may mutate controls, output fragments, context state, `busy`, and input read-only state.

- [ ] **Step 5: Preserve thinking and approval behavior**

Render a changing thinking line in the output view while busy. It shows `FinPilot is thinking · 已思考 3.2s`; when `context_state.status == "compressing"`, show `FinPilot is compressing context · 已思考 3.2s`.

For approval, marshal the callback to the UI thread, pause the thinking animation, show the existing approval table/prompt in a modal `FloatContainer`, then resume after the decision. Reuse the same `ApprovalService` instance for the whole `FinPilotChatApplication` lifetime.

- [ ] **Step 6: Route `chat_command` to the persistent application**

```python
def chat_command(...):
    ...
    chat_app = FinPilotChatApplication(
        user_id=resolved_user_id,
        chat_id=current_chat_id,
        debug=debug_enabled,
        history=FileHistory(str(_history_path())),
        service_factory=_service_factory,
        approval_service=ApprovalService(callback=_prompt_cli_approval),
        slash_handler=_handle_slash_command,
    )
    _render_splash(resolved_user_id, current_chat_id, debug_enabled)
    chat_app.run()
```

Keep `_run_chat` and `_ThinkingStatus` for `ask`, quiet execution, and compatibility tests; `chat` no longer recreates `_prompt_user_input` per turn.

- [ ] **Step 7: Run CLI tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py tests/test_cli.py -q`

Expected: PASS with no hanging worker threads.

- [ ] **Step 8: Commit**

```powershell
git add finpilot/cli_chat.py finpilot/cli.py tests/test_cli_chat.py tests/test_cli.py
git commit -m "feat: 重构 FinPilot 持久化 CLI 对话界面"
```

### Task 5: 完成 Slash 命令、窄终端与全量回归

**Files:**
- Modify: `finpilot/cli_chat.py`
- Modify: `finpilot/cli.py:365-405,602-704,816-840`
- Modify: `tests/test_cli_chat.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md` only if its CLI screenshot/text documents the old transient input behavior

**Interfaces:**
- Consumes: `GlobalContextSnapshotCache` and persistent application from Tasks 3-4
- Produces: final `/status`, `/context`, `/new`, `/resume`, `/clear`, `/exit` behavior

- [ ] **Step 1: Write failing slash-command and resize tests**

```python
def test_new_and_resume_switch_context_snapshots_in_current_process():
    app = build_chat_application(chat_id="chat-a")
    app.snapshot_cache.put("user-1", "chat-a", global_snapshot(used=100, budget=1000))
    app.switch_chat("chat-b")
    assert app.context_state.status == "waiting"
    app.switch_chat("chat-a")
    assert app.context_state.used_tokens == 100


def test_status_prefers_runtime_global_snapshot_over_history_estimate():
    snapshot = cli._build_status_snapshot(
        "user-1",
        "chat-1",
        False,
        runtime_global=global_snapshot(used=140_000, budget=380_000),
    )
    assert snapshot["context_source"] == "runtime global_default"
    assert snapshot["estimated_tokens"] == 140_000
    assert snapshot["effective_input_budget"] == 380_000


def test_context_rows_wrap_without_losing_session_fields():
    app = build_chat_application(terminal_width=52)
    text = fragment_text(app.status_fragments())
    assert "global context" in text
    assert "user=user-1" in text
    assert "chat=chat-1" in text
    assert "debug=off" in text
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py tests/test_cli.py -k "switch_context or runtime_global or wrap_without" -q`

Expected: FAIL on missing snapshot switching and runtime status integration.

- [ ] **Step 3: Complete command integration**

- `/status` and `/context` render into the persistent output area using the active cached snapshot.
- `/new [chat-id]` changes `chat_id`, resets to the cache entry for that key or `waiting`, and preserves previous chat history in storage.
- `/resume <chat-id>` switches to an in-process cached snapshot when present; after a CLI restart it shows `waiting` until the first resumed request completes.
- `/clear` clears only the visible output viewport, not stored chat history or context snapshots.
- `/exit` waits for no active request; while busy it displays `request in progress` rather than abandoning the worker.
- `/debug on|off` updates the second status row immediately.

- [ ] **Step 4: Add responsive status rows**

Use `Dimension` and a width-sensitive fragment renderer. At widths below 72 columns, move usage/status beneath the progress bar and allow the session row to wrap. Do not truncate `user`, `chat`, or `debug`; shorten only the progress bar from 28 cells down to 8 cells.

- [ ] **Step 5: Run all verification commands**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py tests/test_cli.py tests/test_context_budget.py tests/test_context_summarization.py tests/test_safety_integration.py -q`

Expected: PASS.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS. If unrelated pre-existing dirty files cause a failure, record the exact test and prove it reproduces without this plan's staged files before proceeding.

Run: `.venv\Scripts\python.exe -m ruff check finpilot tests`

Expected: `All checks passed!`

Run: `.venv\Scripts\python.exe -m compileall -q finpilot tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors from plan-owned files.

- [ ] **Step 6: Manually verify the terminal**

Run: `.venv\Scripts\finpilot.exe chat --user-id cli-test --chat-id context-preview`

Verify:

- the input frame and two status rows stay visible after submission;
- waiting changes to thinking, then compressing when a global compression event is emitted;
- elapsed time updates without resizing the layout;
- the final bar uses the response's `effective_input_budget` and `estimated_tokens_after`;
- `/status`, `/new`, `/resume`, `/debug on`, approval prompts, and `/exit` remain usable;
- no `source`, model configuration, or endpoint address appears below the input.

- [ ] **Step 7: Commit**

```powershell
git add finpilot/cli_chat.py finpilot/cli.py tests/test_cli_chat.py tests/test_cli.py README.md
git commit -m "feat: 完善 CLI 全局上下文状态与会话命令"
```

Only include `README.md` in the command if it was actually changed. Before committing, run `git diff --cached --name-only` and confirm `.env.example` is absent.
