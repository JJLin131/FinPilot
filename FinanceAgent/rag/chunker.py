from __future__ import annotations

import re

from FinanceAgent.config import settings


class KnowledgeChunker:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def split(self, content: str) -> list[str]:
        normalized = (content or "").strip()
        if not normalized:
            return []
        sections = self._parse_sections(self._strip_front_matter(normalized))
        chunks: list[str] = []
        for path, body in sections:
            prefix = self._heading_path(path)
            body = body.strip()
            if not body:
                continue
            for part in self._split_oversized(body):
                chunks.append(f"{prefix}\n\n{part}".strip() if prefix else part)
        return chunks or self._split_oversized(normalized)

    def _parse_sections(self, content: str) -> list[tuple[list[str], str]]:
        sections: list[tuple[list[str], str]] = []
        heading_stack: list[str] = []
        body_lines: list[str] = []
        current_path: list[str] = []

        for line in content.splitlines():
            match = self.heading_pattern.match(line)
            if match:
                if any(item.strip() for item in body_lines):
                    sections.append((current_path, "\n".join(body_lines)))
                level = len(match.group(1))
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(match.group(2).strip())
                current_path = list(heading_stack)
                body_lines = []
            else:
                body_lines.append(line)
        if any(item.strip() for item in body_lines):
            sections.append((current_path, "\n".join(body_lines)))
        return sections

    def _split_oversized(self, body: str) -> list[str]:
        chunk_size = max(100, 800)
        overlap = max(0, min(100, chunk_size // 2))
        if len(body) <= chunk_size:
            return [body]
        parts: list[str] = []
        start = 0
        while start < len(body):
            end = min(len(body), start + chunk_size)
            parts.append(body[start:end].strip())
            if end >= len(body):
                break
            start = max(start + 1, end - overlap)
        return parts

    def _strip_front_matter(self, content: str) -> str:
        if not content.startswith("---"):
            return content
        end = content.find("\n---", 3)
        return content if end < 0 else content[end + 4 :].strip()

    def _heading_path(self, path: list[str]) -> str:
        return "" if not path else "章节路径: " + " > ".join(path)
