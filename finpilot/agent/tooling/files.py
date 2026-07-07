from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from finpilot.agent.tooling.file_access import FileAccessStore
from finpilot.config import settings
from finpilot.models import GraphState


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: StrictStr = Field(min_length=1, max_length=1000)
    recursive: StrictBool = True
    limit: StrictInt = Field(default=50, ge=1, le=200)


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: StrictStr = Field(min_length=1, max_length=1000)
    max_chars: StrictInt = Field(default=6000, ge=1, le=50000)


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: StrictStr = Field(min_length=1, max_length=1000)
    content: StrictStr = Field(max_length=100000)
    append: StrictBool = False


def register_file_tools(registry) -> None:
    from finpilot.agent.tools import ToolSpec

    registry.register(
        ToolSpec(
            name="list_files",
            description="List files under a configured readable local path.",
            when_to_use="Use when the user asks what files are available in an authorized directory.",
            arguments={"path": "string", "recursive": "boolean, optional", "limit": "integer, optional"},
            args_model=ListFilesArgs,
            executor=_list_files,
        )
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read text from a configured readable local file.",
            when_to_use="Use when the answer needs content from a local file the user has authorized.",
            arguments={"path": "string", "max_chars": "integer, optional"},
            args_model=ReadFileArgs,
            executor=_read_file,
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Write text to a configured writable local file.",
            when_to_use="Use only when the user explicitly asks to create, overwrite, or append a local file.",
            arguments={"path": "string", "content": "string", "append": "boolean, optional"},
            args_model=WriteFileArgs,
            executor=_write_file,
            risk_level="high",
        )
    )


def _store() -> FileAccessStore:
    return FileAccessStore(Path(settings.tool_access_path))


def _list_files(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    del state
    root = _store().require("read", Path(parameters["path"]))
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")
    recursive = bool(parameters.get("recursive", True))
    limit = int(parameters.get("limit") or 50)
    iterator = root.rglob("*") if recursive else root.iterdir()
    files = []
    for path in sorted((item for item in iterator if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        files.append(
            {
                "name": path.relative_to(root).as_posix(),
                "path": str(path.resolve(strict=False)),
                "size": path.stat().st_size,
            }
        )
        if len(files) >= limit:
            break
    return {"path": str(root), "files": files, "truncated": len(files) >= limit}


def _read_file(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    del state
    path = _store().require("read", Path(parameters["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    max_chars = int(parameters.get("max_chars") or 6000)
    text = path.read_text(encoding="utf-8")
    truncated = len(text) > max_chars
    text = text[:max_chars]
    return {
        "content": [
            {
                "source": str(path),
                "title": path.name,
                "text": text,
                "metadata": {"kind": "file", "truncated": truncated},
            }
        ]
    }


def _write_file(state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
    del state
    path = _store().require("write", Path(parameters["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(parameters.get("append", False)) else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(str(parameters["content"]))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "append": bool(parameters.get("append", False)),
    }
