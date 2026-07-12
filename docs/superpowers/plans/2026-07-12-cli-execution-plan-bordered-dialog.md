# CLI Execution Plan Bordered Dialog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the persistent CLI's plain execution-plan text with one bordered dialog containing visually separated DAG node sections.

**Architecture:** Keep `RuntimeProgressViewState` and runtime events unchanged. Add one Prompt Toolkit fragment renderer that owns border width, node separators, labels, wrapping, and status styles; use that renderer for both the live progress area and the persisted history summary.

**Tech Stack:** Python 3.12, Prompt Toolkit formatted-text fragments, pytest, Ruff.

## Global Constraints

- Use one outer `Execution Plan` border, not one panel per node.
- Keep full Task text; Task and Depends may wrap, but status and duration must remain intact.
- Dynamic progress and persisted history must use the same renderer.
- Preserve the stable output-cursor snapshot and existing runtime-event contracts.
- Do not include existing `.env.example` or `finpilot/config.py` changes in the commit.

---

### Task 1: Build a width-aware bordered plan renderer

**Files:**
- Modify: `finpilot/cli_chat.py`
- Test: `tests/test_cli_chat.py`

**Interfaces:**
- Consumes: `RuntimeProgressViewState`, terminal column width, `StyleAndTextTuples`.
- Produces: `_runtime_plan_fragments() -> StyleAndTextTuples`, used by live and persisted output.

- [ ] **Step 1: Write failing border and node-separator tests**

```python
def test_execution_plan_renders_one_bordered_dialog_with_node_sections():
    app = _runtime_app_with_three_nodes()
    rendered = "".join(text for _, text in app._runtime_plan_fragments())

    assert rendered.count("╭─ Execution Plan") == 1
    assert rendered.count("├") == 2
    assert rendered.rstrip().endswith("╰" + "─" * (app.application.output.get_size().columns - 5) + "╯")
    assert "│ Status   ✓ SUCCEEDED" in rendered
    assert "│ Task     Query the user's treasury account" in rendered


def test_execution_plan_border_and_status_survive_narrow_terminal():
    app = _runtime_app_with_long_task()
    fragments = app._runtime_plan_fragments()
    rendered = "".join(text for _, text in fragments)

    assert "SUCCEEDED" in rendered
    assert "S\nUCCEEDED" not in rendered
    assert "╭─ Execution Plan" in rendered
    assert "╰" in rendered and "╯" in rendered
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py::test_execution_plan_renders_one_bordered_dialog_with_node_sections tests/test_cli_chat.py::test_execution_plan_border_and_status_survive_narrow_terminal -q
```

Expected: FAIL because `_runtime_plan_fragments` does not exist and the current output has no closed outer border or node separators.

- [ ] **Step 3: Implement the shared fragment renderer**

Add status metadata and a renderer with this shape:

```python
_RUNTIME_STATUS_PRESENTATION = {
    "PENDING": ("○", "class:context.value"),
    "RUNNING": ("◉", "class:thinking.elapsed"),
    "SUCCEEDED": ("✓", "class:context.ready"),
    "FAILED": ("✗", "class:context.error"),
    "BLOCKED": ("!", "class:context.error"),
    "SKIPPED": ("↷", "class:context.warning"),
}

def _runtime_plan_fragments(self) -> StyleAndTextTuples:
    width = max(40, self.application.output.get_size().columns - 2)
    inner_width = width - 2
    fragments = [("class:system.border", f"╭─ Execution Plan {'─' * max(1, inner_width - 17)}╮\n")]
    for index, node in enumerate(self.runtime_state.plan.get("nodes", [])):
        if index:
            fragments.append(("class:system.border", f"├{'─' * inner_width}┤\n"))
        fragments.extend(self._runtime_node_fragments(node, inner_width))
    fragments.append(("class:system.border", f"╰{'─' * inner_width}╯\n"))
    return fragments
```

Implement `_runtime_node_fragments()` so every physical line begins with a styled `│ ` border; status and duration share their own line, while `Node`, `Agent`, `Task`, and `Depends` are separate labeled fields. Use `prompt_toolkit.formatted_text.utils.split_lines` or an equivalent cell-width-aware helper to wrap values without breaking the outer border.

- [ ] **Step 4: Run renderer tests and confirm GREEN**

Run the Step 2 command again. Expected: `2 passed`.

---

### Task 2: Reuse the bordered renderer for live and persisted output

**Files:**
- Modify: `finpilot/cli_chat.py`
- Test: `tests/test_cli_chat.py`

**Interfaces:**
- Consumes: `_runtime_plan_fragments()` from Task 1.
- Produces: identical bordered structure in `_runtime_progress_fragments()` and `_persist_runtime_summary()`.

- [ ] **Step 1: Write the failing reuse test**

```python
def test_live_and_persisted_execution_plan_share_bordered_structure():
    app = _runtime_app_with_three_nodes()
    live = "".join(text for _, text in app._runtime_progress_fragments())
    app._persist_runtime_summary()
    history = "".join(text for _, text in app._history_fragments)

    assert "╭─ Execution Plan" in live
    assert "╭─ Execution Plan" in history
    assert live.count("├") == history.count("├") == 2
```

- [ ] **Step 2: Run the test and confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py::test_live_and_persisted_execution_plan_share_bordered_structure -q
```

Expected: FAIL because persisted output still rebuilds plan content through `_append_panel()`.

- [ ] **Step 3: Connect both paths to the shared renderer**

In `_runtime_progress_fragments()`, replace the `_runtime_plan_lines()` loop with:

```python
if self.runtime_state.plan:
    fragments.extend(self._runtime_plan_fragments())
```

In `_persist_runtime_summary()`, append a copy of `_runtime_plan_fragments()` directly to `_history_fragments`, add the Planner duration inside the same outer dialog before its closing border, and update `output_text` from the fragment text. Remove `_runtime_plan_lines()` and `_truncate_cells()` when no callers remain.

- [ ] **Step 4: Run CLI tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli_chat.py tests/test_cli.py -q
```

Expected: all CLI tests pass, including long-task, cursor snapshot, runtime callback, approval, and persistent-application tests.

- [ ] **Step 5: Run full verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q finpilot
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 6: Commit only scoped files with Chinese notes**

```powershell
git add -- finpilot/cli_chat.py tests/test_cli_chat.py
git diff --cached --check
git commit -m "feat: 使用边框展示执行计划" -m "将动态与历史执行计划统一为单外框节点分区布局，并保持长任务与窄屏下的状态可读性。"
```
