from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

INCLUDE_RE = re.compile(r'\\include\s+"([^"]+)"', re.I)

class FileResolver:
    def __init__(self, base_dir: Path, max_depth: int = 10, exclude_pattern: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.max_depth = max_depth
        self.exclude_pattern = re.compile(exclude_pattern, re.I) if exclude_pattern else None
        self.resolved_cache = {}
        self.in_progress = set()

    def _find_include_file(self, include_path: str) -> Optional[Path]:
        for base in [self.base_dir, self.base_dir.parent]:
            candidate = base / include_path
            if candidate.exists():
                return candidate

        include_norm = unicodedata.normalize("NFC", include_path)
        if include_norm != include_path:
            for base in [self.base_dir, self.base_dir.parent]:
                candidate = base / include_norm
                if candidate.exists():
                    return candidate
        return None

    def _should_skip_include(self, include_path: str) -> bool:
        return bool(self.exclude_pattern and self.exclude_pattern.search(include_path))

    def _resolve_recursive(self, file_path: Path, depth: int = 0) -> str:
        if depth > self.max_depth:
            print(f"warning: Max recursion depth ({self.max_depth}) reached for {file_path}", file=sys.stderr)
            return file_path.read_text(encoding="utf-8", errors="ignore")

        file_key = str(file_path.resolve())

        if file_key in self.resolved_cache:
            return self.resolved_cache[file_key]

        if file_key in self.in_progress:
            print(f"warning: Circular include detected: {file_path}", file=sys.stderr)
            return ""

        self.in_progress.add(file_key)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            def replace_include(match: re.Match) -> str:
                include_path = match.group(1)
                if self._should_skip_include(include_path):
                    return match.group(0)

                resolved_file = self._find_include_file(include_path)
                if not resolved_file:
                    include_name = Path(include_path).name
                    if include_name.lower().endswith(".ly"):
                        lang = include_name.rsplit(".", 1)[0].lower()
                        lang_map = {"italiano", "english", "deutsch", "francais", "espanol", "nederlands", "norsk", "suomi", "svenska", "vlaams"}
                        if lang in lang_map:
                            return f'\\language "{lang}"'
                    print(f"warning: Include file not found: {include_path} (from {file_path})", file=sys.stderr)
                    return match.group(0)

                return self._resolve_recursive(resolved_file, depth + 1)

            resolved = INCLUDE_RE.sub(replace_include, content)
            self.resolved_cache[file_key] = resolved
            return resolved

        finally:
            self.in_progress.discard(file_key)

    def resolve(self, file_path: Path) -> str:
        self.resolved_cache.clear()
        self.in_progress.clear()
        return self._resolve_recursive(file_path)


def run(file_path: Path, base_dir: Optional[Path] = None, exclude_variabili: bool = False, split_forma: bool = True) -> list[str]:
    base_dir = base_dir or file_path.parent
    exclude_pattern = "variabili" if exclude_variabili else None
    resolver = FileResolver(base_dir, exclude_pattern=exclude_pattern)
    resolved_text = resolver.resolve(file_path)
    return split_on_multiple_forma(resolved_text) if split_forma else [resolved_text]


def split_on_multiple_forma(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^forma\s*=\s*\{", text))
    if len(matches) <= 1:
        return [text]

    def _find_matching_brace(source: str, open_index: int) -> int:
        depth = 0
        for i in range(open_index, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    prefix = text[:matches[0].start()]
    last_close = _find_matching_brace(text, matches[-1].end() - 1)
    suffix = text[last_close + 1:] if last_close != -1 else ""

    pieces = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pieces.append(prefix + text[start:end] + suffix)

    return [re.sub(r'(?m)^\s*\\version\s+"[^"]+"\s*\r?\n?', "", p).lstrip() for p in pieces]
