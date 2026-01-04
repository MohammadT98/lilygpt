from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Set


INCLUDE_RE = re.compile(r'\\include\s+"([^"]+)"', re.I)


class FileResolver:
    def __init__(self, base_dir: Path, max_depth: int = 10, exclude_pattern: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.max_depth = max_depth
        self.exclude_pattern = re.compile(exclude_pattern, re.I) if exclude_pattern else None
        self.resolved_cache: dict[str, str] = {}
        self.in_progress: Set[str] = set()

    def _find_include_file(self, include_path: str) -> Optional[Path]:
        candidate = self.base_dir / include_path
        if candidate.exists():
            return candidate

        alt = self.base_dir.parent / include_path
        if alt.exists():
            return alt

        include_norm = unicodedata.normalize("NFC", include_path)
        if include_norm != include_path:
            candidate_norm = self.base_dir / include_norm
            if candidate_norm.exists():
                return candidate_norm
            alt_norm = self.base_dir.parent / include_norm
            if alt_norm.exists():
                return alt_norm

        return None

    def _should_skip_include(self, include_path: str) -> bool:
        if self.exclude_pattern and self.exclude_pattern.search(include_path):
            return True
        return False

    def _resolve_recursive(self, file_path: Path, depth: int = 0) -> str:
        if depth > self.max_depth:
            print(
                f"warning: Max recursion depth ({self.max_depth}) reached for {file_path}",
                file=sys.stderr,
            )
            return file_path.read_text(encoding="utf-8", errors="ignore")

        file_key = str(file_path.resolve())

        if file_key in self.resolved_cache:
            return self.resolved_cache[file_key]

        if file_key in self.in_progress:
            print(
                f"warning: Circular include detected: {file_path}",
                file=sys.stderr,
            )
            return ""

        self.in_progress.add(file_key)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            def replace_include(match: re.Match) -> str:
                include_path = match.group(1)

                if self._should_skip_include(include_path):
                    return match.group(0)

                resolved_file = self._find_include_file(include_path)
                include_name = Path(include_path).name
                if not resolved_file:
                    language_stem = include_name.rsplit(".", 1)[0]
                    if include_name.lower().endswith(".ly"):
                        lang_map = {
                            "italiano": "italiano",
                            "english": "english",
                            "deutsch": "deutsch",
                            "francais": "francais",
                            "espanol": "espanol",
                            "nederlands": "nederlands",
                            "norsk": "norsk",
                            "suomi": "suomi",
                            "svenska": "svenska",
                            "vlaams": "vlaams",
                        }
                        lang = lang_map.get(language_stem.lower())
                        if lang:
                            return f'\\language "{lang}"'
                    print(
                        f"warning: Include file not found: {include_path} (from {file_path})",
                        file=sys.stderr,
                    )
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


def run(
    file_path: Path,
    base_dir: Optional[Path] = None,
    exclude_variabili: bool = False,
    split_forma: bool = True,
) -> list[str]:
    if base_dir is None:
        base_dir = file_path.parent

    exclude_pattern = "variabili" if exclude_variabili else None
    resolver = FileResolver(base_dir, exclude_pattern=exclude_pattern)

    resolved_text = resolver.resolve(file_path)

    if split_forma:
        return split_on_multiple_forma(resolved_text)
    else:
        return [resolved_text]


def split_on_multiple_forma(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^forma\s*=\s*\{", text))
    if len(matches) <= 1:
        return [text]

    def _find_matching_brace(source: str, open_index: int) -> int:
        depth = 0
        for idx in range(open_index, len(source)):
            char = source[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    prefix = text[: matches[0].start()]
    pieces: list[str] = []

    last_match = matches[-1]
    last_open_index = last_match.end() - 1
    last_close_index = _find_matching_brace(text, last_open_index)
    suffix_after_last = text[last_close_index + 1 :] if last_close_index != -1 else ""

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        pieces.append(prefix + body + suffix_after_last)

    normalized: list[str] = []
    for piece in pieces:
        cleaned = re.sub(
            r'(?m)^\s*\\version\s+"[^"]+"\s*\r?\n?',
            "",
            piece,
        )
        normalized.append(cleaned.lstrip())

    return normalized
