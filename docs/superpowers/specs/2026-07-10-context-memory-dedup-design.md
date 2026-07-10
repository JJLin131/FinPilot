# Context Memory Deduplication Design

## Goal

Remove the derived `long_term_memory` field while preserving the existing meaning of memory fields:

- `recent_messages`: short-term messages from the current chat.
- `structured_memory`: structured user profile loaded from MySQL.
- `semantic_memory`: relevant long-term memories recalled from Chroma.
- `LoopContext`: working memory for the current LangGraph run.

## Context Ownership

`GraphState` remains the complete runtime state. `SessionContext` remains the user/session projection used by agents. Neither model stores a second formatted copy of structured and semantic memory.

The following field is removed from `MemoryContext`, `GraphState`, and `SessionContext`:

```python
long_term_memory: list[str]
```

`MemoryManager.load()` returns the original structured and semantic records without calling `_format_long_term_memory()`.

## Prompt Flow

The decision prompt continues to consume the existing `SessionContext + SubAgentContext + LoopContext` bundle.

The final-answer prompt consumes a bounded projection containing:

```text
user_message
recent_messages
structured_memory
semantic_memory
evidence
working_notes
```

The final-answer path must not rebuild or append a formatted `long_term_memory` string. `user_id`, `chat_id`, `memory_id`, request identifiers, and trace identifiers remain runtime metadata and are not required in the final-answer prompt.

## Compatibility

This is an internal model and prompt refactor. It does not change MySQL tables, Chroma collections, memory extraction, semantic merge behavior, or stored memory data.

Serialized debug state may stop containing `long_term_memory`; existing consumers should use `structured_memory` and `semantic_memory` instead.

## Verification

Tests must prove that:

- Chroma recall still populates `semantic_memory`.
- MySQL profile loading still populates `structured_memory`.
- `long_term_memory` is absent from runtime and session models.
- Decision prompts still receive recent, structured, and semantic memory.
- Final-answer prompts receive recent, structured, and semantic memory exactly once.
- The complete test suite remains green.
