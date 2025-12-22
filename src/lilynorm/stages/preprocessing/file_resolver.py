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

    resolved = resolver.resolve(file_path)
    
    # Fix common typo: set-defaultpaper-size -> set-default-paper-size
    resolved = resolved.replace("set-defaultpaper-size", "set-default-paper-size")
    
    # Remove LilyPond line continuation markers
    # Raw source patterns:
    #   re4.-+ mib8        (hyphen-plus at line continuation)
    #   re4.-\cmd -+ mib8  (hyphen before command AND hyphen-plus)
    # After removing: should be re4. and re4. \cmd
    import re
    resolved = re.sub(r'-\+', ' ', resolved)      # Remove -+ (replace with space)
    resolved = re.sub(r'-(?=\\)', ' ', resolved)   # Remove - before \ (replace with space)

    # Keep only the first \version declaration to avoid duplicates after inlining includes
    version_seen = False

    def _keep_first_version(match: re.Match) -> str:
        nonlocal version_seen
        if version_seen:
            return ""
        version_seen = True
        return match.group(0)

    resolved = re.sub(r'(^|\n)\s*\\version\s+"[^"]+"\s*', _keep_first_version, resolved, flags=re.MULTILINE)
    
    return resolved


def split_on_multiple_forma(text: str) -> list[str]:
    """Split a resolved LilyPond file into pieces if it defines multiple top-level `forma = { ... }` blocks.

    The shared header (comments, \version, \language, includes already inlined) is kept in each piece to keep
    outputs standalone. If only one `forma` exists, the original text is returned in a single-element list.
    """
    matches = list(re.finditer(r"(?m)^forma\s*=\s*\{", text))
    if len(matches) <= 1:
        return [text]

    prefix = text[: matches[0].start()]
    pieces: list[str] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        pieces.append(prefix + body)

    return pieces
