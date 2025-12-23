"""
LilyPond file resolution: resolves \\include statements and inlines dependencies.

Converts modular LilyPond files with \\include directives into self-contained
standalone files by recursively inlining all included files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Set


# Regex to match \include "filename" statements
INCLUDE_RE = re.compile(r'\\include\s+"([^"]+)"', re.I)


class FileResolver:
    """Resolves \\include statements in LilyPond files."""

    def __init__(self, base_dir: Path, max_depth: int = 10, exclude_pattern: Optional[str] = None):
        """
        Initialize the resolver.

        Args:
            base_dir: Root directory to search for included files.
            max_depth: Maximum recursion depth to prevent infinite loops.
            exclude_pattern: Regex pattern for files to skip inlining (e.g., "variabili").
        """
        self.base_dir = Path(base_dir)
        self.max_depth = max_depth
        self.exclude_pattern = re.compile(exclude_pattern, re.I) if exclude_pattern else None
        self.resolved_cache: dict[str, str] = {}
        self.in_progress: Set[str] = set()

    def _find_include_file(self, include_path: str) -> Optional[Path]:
        """Find the actual file for an include statement."""
        # Try relative to base_dir
        candidate = self.base_dir / include_path
        if candidate.exists():
            return candidate

        # Try in the same directory as base_dir
        alt = self.base_dir.parent / include_path
        if alt.exists():
            return alt

        return None

    def _should_skip_include(self, include_path: str) -> bool:
        """Return True if this include should not be inlined (e.g., variabili)."""
        if self.exclude_pattern and self.exclude_pattern.search(include_path):
            return True
        return False

    def _resolve_recursive(self, file_path: Path, depth: int = 0) -> str:
        """
        Recursively resolve includes in a file.

        Args:
            file_path: Path to the LilyPond file.
            depth: Current recursion depth.

        Returns:
            The file content with includes resolved.
        """
        if depth > self.max_depth:
            print(
                f"WARNING: Max recursion depth ({self.max_depth}) reached for {file_path}",
                file=sys.stderr,
            )
            return file_path.read_text(encoding="utf-8", errors="ignore")

        file_key = str(file_path.resolve())

        # Check cache
        if file_key in self.resolved_cache:
            return self.resolved_cache[file_key]

        # Detect circular includes
        if file_key in self.in_progress:
            print(
                f"WARNING: Circular include detected: {file_path}",
                file=sys.stderr,
            )
            return ""

        self.in_progress.add(file_key)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Find all include statements and replace them
            def replace_include(match: re.Match) -> str:
                include_path = match.group(1)

                # Skip excluded patterns (e.g., variabili)
                if self._should_skip_include(include_path):
                    return match.group(0)  # Keep the include as-is

                # Find the actual file
                resolved_file = self._find_include_file(include_path)
                if not resolved_file:
                    print(
                        f"WARNING: Include file not found: {include_path} (from {file_path})",
                        file=sys.stderr,
                    )
                    return match.group(0)  # Keep the include as-is

                # Recursively resolve the included file
                included_content = self._resolve_recursive(resolved_file, depth + 1)
                return included_content

            resolved = INCLUDE_RE.sub(replace_include, content)
            self.resolved_cache[file_key] = resolved
            return resolved

        finally:
            self.in_progress.discard(file_key)

    def resolve(self, file_path: Path) -> str:
        """
        Resolve all includes in a file and return the self-contained content.

        Args:
            file_path: Path to the LilyPond file.

        Returns:
            File content with all (non-excluded) includes inlined.
        """
        self.resolved_cache.clear()
        self.in_progress.clear()
        return self._resolve_recursive(file_path)


def run(
    text: str,
    file_path: Path,
    base_dir: Optional[Path] = None,
    exclude_variabili: bool = False,
) -> str:
    """
    Resolve \\include statements in LilyPond text.

    Args:
        text: The LilyPond file content.
        file_path: The path to the LilyPond file (used to infer base_dir).
        base_dir: Base directory for resolving includes. Defaults to file_path.parent.
        exclude_variabili: If True, keep \\include "...variabili..." as-is instead of inlining.

    Returns:
        The LilyPond content with includes resolved (except excluded ones).
    """
    if base_dir is None:
        base_dir = file_path.parent

    exclude_pattern = "variabili" if exclude_variabili else None
    resolver = FileResolver(base_dir, exclude_pattern=exclude_pattern)

    return resolver.resolve(file_path)


def split_on_multiple_forma(text: str) -> list[str]:
    """Split a resolved LilyPond file into pieces if it defines multiple top-level `forma = { ... }` blocks.

    The shared header (comments, \version, \language, includes already inlined) is kept in each piece to keep
    outputs standalone. If only one `forma` exists, the original text is returned in a single-element list.
    """
    # Normalize version declarations to a single 2.24.4 header.
    text = re.sub(r'(?m)^\\version\\s+"[^"]+"\\s*\\n?', "", text)
    text = '\\version "2.24.4"\\n' + text.lstrip()

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
        # Keep shared tail content (e.g., score blocks) for every piece.
        pieces.append(prefix + body + suffix_after_last)

    return pieces
