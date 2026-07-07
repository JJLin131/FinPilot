from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AccessMode = Literal["read", "write"]


@dataclass(frozen=True)
class AccessRule:
    path: Path
    recursive: bool = True

    def model_dump(self) -> dict[str, object]:
        return {"path": str(self.path), "recursive": self.recursive}


class FileAccessStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, list[AccessRule]]:
        if not self.path.exists():
            return {"read": [], "write": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "read": [self._rule(item) for item in payload.get("read", []) if isinstance(item, dict)],
            "write": [self._rule(item) for item in payload.get("write", []) if isinstance(item, dict)],
        }

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        rules = self.load()
        return {
            "read": [rule.model_dump() for rule in rules["read"]],
            "write": [rule.model_dump() for rule in rules["write"]],
        }

    def allow(self, mode: AccessMode, path: Path, *, recursive: bool = True) -> None:
        rules = self.load()
        resolved = self._resolve(path)
        rule = AccessRule(path=resolved, recursive=recursive)
        kept = [
            existing
            for existing in rules[mode]
            if not (_same_path(existing.path, rule.path) and existing.recursive == rule.recursive)
        ]
        rules[mode] = kept + [rule]
        self._save(rules)

    def revoke(self, mode: AccessMode, path: Path) -> bool:
        rules = self.load()
        resolved = self._resolve(path)
        before = len(rules[mode])
        rules[mode] = [rule for rule in rules[mode] if not _same_path(rule.path, resolved)]
        self._save(rules)
        return len(rules[mode]) != before

    def require(self, mode: AccessMode, path: Path) -> Path:
        resolved = self._resolve(path)
        if not self.is_allowed(mode, resolved):
            raise PermissionError(f"Path is not authorized for {mode}: {resolved}")
        return resolved

    def is_allowed(self, mode: AccessMode, path: Path) -> bool:
        resolved = self._resolve(path)
        return any(_matches(rule, resolved) for rule in self.load()[mode])

    def _save(self, rules: dict[str, list[AccessRule]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "read": [rule.model_dump() for rule in rules["read"]],
            "write": [rule.model_dump() for rule in rules["write"]],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rule(self, item: dict) -> AccessRule:
        return AccessRule(path=self._resolve(Path(str(item["path"]))), recursive=bool(item.get("recursive", True)))

    def _resolve(self, path: Path) -> Path:
        return path.expanduser().resolve(strict=False)


def _matches(rule: AccessRule, target: Path) -> bool:
    if _same_path(rule.path, target):
        return True
    if rule.recursive:
        return any(_same_path(parent, rule.path) for parent in target.parents)
    return _same_path(target.parent, rule.path)


def _same_path(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(str(left))
    right_text = os.path.normcase(str(right))
    return left_text == right_text

