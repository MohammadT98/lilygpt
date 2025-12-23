from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable


# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------

# If enabled, drop assignments that resolve to empty blocks (e.g. foo = {}).
DROP_EMPTY_ASSIGNMENTS = False

# If enabled, remove spacer-only subvoices (e.g. "\\\\ { s1 s1 }").
PRUNE_SPACER_SUBVOICES = True

# Controls whitespace compaction strategy ("safe" or "simple").
DEFAULT_SPACE_MODE = "safe"

# Toggle removal of \score/\layout/\midi wrappers (training-only noise).
# Set to False to keep layout blocks for PDF generation/verification.
# Set to True for pure ML training to maximize noise removal.
# Can be overridden by LILYNORM_KEEP_LAYOUT environment variable.
STRIP_SCORE_LAYOUT = True if not os.environ.get("LILYNORM_KEEP_LAYOUT") else False


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _grab_balanced(
    text: str,
    start: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int:
    """
    Given an index `start` pointing at an opening brace (or other delimiter),
    return the index of the matching closing delimiter, handling nested pairs.

    Returns -1 if the text ends before a matching closing char is found.
    """
    depth = 1
    index = start + 1
    length = len(text)

    while index < length:
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1

    return -1


def _protect_scheme_expressions(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace #(...) Scheme expressions with placeholders to protect them from regex processing.
    Handles nested parentheses. Returns (protected_text, mapping_dict).
    """
    import hashlib
    protected = {}
    result_parts = []
    i = 0
    
    while i < len(text):
        # Look for #(
        if i < len(text) - 1 and text[i:i+2] == '#(':
            # Find the matching closing paren
            paren_depth = 1
            j = i + 2
            while j < len(text) and paren_depth > 0:
                if text[j] == '(':
                    paren_depth += 1
                elif text[j] == ')':
                    paren_depth -= 1
                j += 1
            
            # Extract the full Scheme expression
            scheme_expr = text[i:j]
            # Use SCHEMEX format without underscores (they might get cleaned up by other regexes)
            key = f"SCHEMEX{hashlib.md5(scheme_expr.encode()).hexdigest()[:8]}X"
            protected[key] = scheme_expr
            result_parts.append(key)
            i = j
        else:
            result_parts.append(text[i])
            i += 1
    
    return ''.join(result_parts), protected


def _restore_scheme_expressions(text: str, mapping: Dict[str, str]) -> str:
    """Restore protected Scheme expressions."""
    for key, original in mapping.items():
        text = text.replace(key, original)
    return text


def _remove_block_directive(src: str, directive: str) -> Tuple[str, int]:
    """
    Remove top-level \\<directive> { ... } blocks from the given LilyPond source.

    Returns a tuple of (new_source, removed_count).
    """
    pattern = re.compile(rf"\\{directive}\s*\{{", re.M)
    removed_count = 0
    output_parts: List[str] = []
    search_start = 0

    while True:
        match = pattern.search(src, search_start)
        if not match:
            output_parts.append(src[search_start:])
            break

        output_parts.append(src[search_start:match.start()])

        brace_open_index = match.end() - 1
        brace_close_index = _grab_balanced(src, brace_open_index, "{", "}")

        # If we cannot find a matching closing brace, keep the directive as-is.
        if brace_close_index == -1:
            output_parts.append(src[match.start():match.end()])
            search_start = match.end()
            continue

        removed_count += 1
        search_start = brace_close_index + 1

    return "".join(output_parts), removed_count


def _remove_top_level_scheme_blocks(src: str) -> Tuple[str, int]:
    """Remove top-level Scheme blocks that start with '#(' on a line.

    Uses balanced parentheses via _grab_balanced to safely remove nested forms.
    Returns (new_source, removed_count).
    """
    removed_count = 0
    output_parts: List[str] = []
    i = 0
    while i < len(src):
        # Find start of a line
        line_start = i
        # Advance to end of line or string
        while i < len(src) and src[i] not in "\n":
            i += 1
        line = src[line_start:i]

        # Check for top-level scheme start
        m = re.match(r"^\s*#\(", line)
        if m:
            # Compute absolute position of '(' in src
            paren_open = line_start + m.end() - 1
            paren_close = _grab_balanced(src, paren_open, "(", ")")
            if paren_close != -1:
                # Skip entire scheme block and following newline
                removed_count += 1
                i = paren_close + 1
                # Also skip a single trailing newline if present
                if i < len(src) and src[i] == "\n":
                    i += 1
                # Replace with nothing by not appending this line
                continue
        # Keep original line including newline
        output_parts.append(src[line_start:i])
        if i < len(src) and src[i] == "\n":
            output_parts.append("\n")
            i += 1

    return "".join(output_parts), removed_count


def _remove_with_blocks(src: str) -> Tuple[str, int]:
    """
    Remove \\with { ... } blocks from the given LilyPond source.

    Returns a tuple of (new_source, removed_count).
    """
    pattern = re.compile(r"\\with\s*\{", re.M)
    removed_count = 0
    output_parts: List[str] = []
    search_start = 0

    while True:
        match = pattern.search(src, search_start)
        if not match:
            output_parts.append(src[search_start:])
            break

        output_parts.append(src[search_start:match.start()])

        brace_open_index = match.end() - 1
        brace_close_index = _grab_balanced(src, brace_open_index, "{", "}")

        # If we cannot find a matching closing brace, keep the original text.
        if brace_close_index == -1:
            output_parts.append(src[match.start():match.end()])
            search_start = match.end()
            continue

        removed_count += 1
        search_start = brace_close_index + 1

    return "".join(output_parts), removed_count


def _strip_lyricmode_assignments(text: str) -> Tuple[str, int]:
    """
    Remove `name = \\lyricmode { ... }` assignments entirely.
    Empty assignments will be cleaned up by _remove_empty_variable_assignments.

    Returns (new_text, removed_count).
    """
    removed_count = 0

    while True:
        match = RE_LYRIC_ASSIGN.search(text)
        if not match:
            break

        brace_open_index = match.end() - 1
        brace_close_index = _grab_balanced(text, brace_open_index, "{", "}")
        if brace_close_index == -1:
            break

        # Find the variable name to also remove references
        var_name_match = re.match(r'^\s*([A-Za-z_@][\w@]*)\s*=', match.group(0))
        var_name = var_name_match.group(1) if var_name_match else None
        
        # Remove the entire assignment (including trailing newline)
        assignment_end = brace_close_index + 1
        if assignment_end < len(text) and text[assignment_end] == '\n':
            assignment_end += 1
        
        text = text[:match.start()] + text[assignment_end:]
        
        # Remove references to this variable
        if var_name:
            ref_pattern = re.compile(rf"\\{re.escape(var_name)}\b")
            text = ref_pattern.sub("", text)
        
        removed_count += 1

    return text, removed_count


def _strip_inline_lyricmode(text: str) -> Tuple[str, int]:
    """
    Replace inline `\\lyricmode { ... }` with `{}` while preserving
    surrounding whitespace as much as possible.

    Returns (new_text, removed_count).
    """
    removed_count = 0
    search_start = 0
    output_parts: List[str] = []

    while True:
        match = RE_LYRIC_INLINE.search(text, search_start)
        if not match:
            output_parts.append(text[search_start:])
            break

        literal_start = match.start()
        brace_open_index = match.end() - 1
        brace_close_index = _grab_balanced(text, brace_open_index, "{", "}")
        if brace_close_index == -1:
            output_parts.append(text[search_start:match.end()])
            search_start = match.end()
            continue

        prefix = text[search_start:literal_start]
        trimmed_prefix = prefix.rstrip(" \t")
        output_parts.append(trimmed_prefix)
        output_parts.append("{}")

        removed_count += 1
        search_start = brace_close_index + 1

    return "".join(output_parts), removed_count


def _remove_custom_assignments(text: str) -> Tuple[str, int]:
    """Remove dataset-specific helper assignments like `notrasp = ...` or `forma = {...}`.

    Handles both single-line definitions and balanced-brace blocks. Returns (new_text, removed_count).
    """
    removed = 0
    for name in CUSTOM_ASSIGNMENT_NAMES:
        # Match the assignment start; RHS can be token, string, or brace block.
        pattern = re.compile(rf"(?m)^\s*{re.escape(name)}\s*=\s*")
        search_start = 0

        while True:
            m = pattern.search(text, search_start)
            if not m:
                break

            assign_start = m.end()
            # Skip whitespace after the equals sign
            while assign_start < len(text) and text[assign_start] in " \t":
                assign_start += 1

            removal_end = assign_start
            if assign_start < len(text) and text[assign_start] == "{":
                close_idx = _grab_balanced(text, assign_start, "{", "}")
                if close_idx != -1:
                    removal_end = close_idx + 1
                else:
                    # Fallback: drop to end of line if braces are unbalanced
                    nl = text.find("\n", assign_start)
                    removal_end = len(text) if nl == -1 else nl
            else:
                nl = text.find("\n", assign_start)
                removal_end = len(text) if nl == -1 else nl

            # Also remove trailing newline if present
            if removal_end < len(text) and text[removal_end] == "\n":
                removal_end += 1

            text = text[: m.start()] + text[removal_end:]
            removed += 1
            search_start = m.start()

    return text, removed


def _remove_metadata_headers(text: str) -> Tuple[str, int]:
    """Drop header-style metadata lines that add noise for training (keep \language for pitch names)."""
    pattern = re.compile(r"(?m)^\s*\\version\b.*$")
    cleaned, removed = pattern.subn("", text)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Regex definitions
# ---------------------------------------------------------------------------

RE_OVERRIDES = [
    re.compile(r"(?:\\once\s+)?\\override\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\revert\b[^\s{}]+", re.I),
    re.compile(r"(?:\\once\s+)?(?:[-_^]\s*)?\\tweak\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\shape\b[^\n\r{}]*", re.I),
    re.compile(r"(?:\\once\s+)?\\(?:undo\s+)?omit\b[^\s{}]+", re.I),
    re.compile(
        r"(?:\\once\s+)?\\(?:hideNotes|magnifyStaff|teeny|tiny|small|large|huge)\b"
        r"(?:[^\S\n][^\n\r{}]*)?"
    ),
        # Slur/tie style toggles commonly used with \\once
        re.compile(r"(?:\\once\s+)?\\(?:slurDashed|slurSolid|tieDashed|tieSolid)\b", re.I),
        # Match variants like: \change Staff, \change Staff = "down", \change Staff = #'up
    # Stem, slur, tie, beam directions and other visual overrides
    re.compile(r"(?:\\once\s+)?\\(?:stemUp|stemDown|stemNeutral|slurUp|slurDown|slurNeutral|tieUp|tieDown|tieNeutral|shiftOn|shiftOff|shiftOnn|shiftOnnn)\b", re.I),
    # Match variants like: \change Staff, \change Staff = "down", \change Staff = #'up
    re.compile(r"\\change\s+Staff(?:\s*=\s*(?:\"[^\"]*\"|#'?[A-Za-z]+))?", re.I),
]

RE_MARKUP = re.compile(r"(?:[-_^]\s*)?\\markup\b", re.I)
RE_MARK = re.compile(r"(?:[-_^]\s*)?\\mark\b", re.I)

# Performance instructions (soli, tutti, etc.) - treated as markup
RE_PERFORMANCE_MARKS = re.compile(r"\\(?:soli|tu|solo|tutti)\b", re.I)

# Dataset-specific custom commands that should be removed
RE_CUSTOM_COMMANDS = re.compile(
    r"\\(?:terzine|con|senza|mbreak|trasp|notrasp|typeset|notypeset)\\b",
    re.I
)

CUSTOM_ASSIGNMENT_NAMES = (
    "terzine",
    "con",
    "senza",
    "mbreak",
    "trasp",
    "notrasp",
    "typeset",
    "notypeset",
)

DYNAMICS = (
    "ppppp|pppp|ppp|pp|p|mp|mf|f|ff|fff|ffff|fffff|fp|sf|sfz|sffz|rfz|fz|sfp|sff|sfpp|sfzp"
)
RE_DYNAMICS = re.compile(rf"(?:[-_^]\s*)?\\(?:{DYNAMICS})\b", re.I)

RE_HAIRPINS = re.compile(
    r"\\[<>!]|\\(?:cresc|decresc|decr|dim|crescendo|diminuendo)\b", re.I
)
RE_ATTACHED_QUOTES = re.compile(r"(?:[-_^]\s*)\"[^\"]*\"")
# Quotes attached directly to a token (e.g., r8"Sempre piano")
RE_INLINE_QUOTES = re.compile(r'(?<=\S)"[^"\n]*"')
RE_LYRIC_ASSIGN = re.compile(
    r"(?m)(^\s*[A-Za-z_@][\w@]*\s*=\s*)\\lyricmode\s*\{"
)
RE_LYRIC_INLINE = re.compile(r"\\lyricmode\s*\{", re.I)

HSPACE = re.compile(r"[ \t]+")
RE_STRAY_ATTACH = re.compile(
    r"(?m)([-_^])(?=\s*(?:$|[\r\n]|[,;:|)}\]]|(?!(?:\\|\"|\{|\<|\>|\!|[a-gris][',]*))))"
)
# Remove stray \once tokens that are not followed by a Lily command
RE_LONE_ONCE = re.compile(r"(?m)\\once\b(?:[ \t]+(?=$|[\r\n}%])|[ \t]*(?!\\[A-Za-z]))")
RE_SPACE_BEFORE_CLOSER = re.compile(r"[ \t]+(?=[)\]}])")
RE_SPACE_AFTER_OPENER = re.compile(r"(?<=[({\[])[ \t]+")
RE_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[,;:|>])")
RE_NOTE_OCTAVE_SPACE = re.compile(r"(?i)(?<=\b[a-gr])\s+(?=[',])")
RE_MULTI_BLANKS = re.compile(r"\n{3,}")

RE_ASSIGN_OPEN = re.compile(r"(?m)^(\s*\w+\s*=\s*)\{\s*$")
RE_EMPTY_BLOCK_LINE = re.compile(r"(?m)^\s*\{\s*\}\s*$")
RE_EMPTY_ASSIGNMENT_LINE = re.compile(
    r"(?m)^\s*[A-Za-z_@][\w@]*\s*=\s*(?:\{\s*\})?\s*$"
)
RE_INLINE_EMPTY_BRACES = re.compile(r"\{\s*\}")
RE_INCLUDE_TAG = re.compile(r"(?m)^\s*<<\s*\\\s*@\w+\b.*?>>\s*$")
RE_REPEATED_INCLUDE = re.compile(r"(<<\s*\\\s*@\w+\b.*?>>\s*)+", re.S)
RE_EMPTY_SCORES = re.compile(
    r"(?ms)^\\score\s*\{\s*\{\s*\}\s*(?:\\layout\s*\{.*?\}\s*)?\}\s*$"
)
RE_EMPTY_LAYOUT_BLOCK = re.compile(r"(?ms)^\\layout\s*\{\s*\}$")


def _collapse_empty_assignment_blocks(text: str) -> str:
    """
    Normalize empty assignment blocks like:

        foo = {
        }

    into a compact single line:

        foo = {}
    
    Also handles single-line empty blocks and nested empty blocks like:
        foo = {}
        foo = { {} }
    """
    index = 0
    output_parts: List[str] = []
    length = len(text)

    while index < length:
        match = RE_ASSIGN_OPEN.search(text, index)
        if not match:
            output_parts.append(text[index:])
            break

        output_parts.append(text[index:match.start()])

        assignment_prefix = match.group(1)
        brace_open_index = match.end() - 1
        brace_close_index = _grab_balanced(text, brace_open_index, "{", "}")

        if brace_close_index == -1:
            # Could not find closing brace; keep up to line end intact.
            line_end_index = text.find("\n", match.end())
            if line_end_index == -1:
                line_end_index = length
            output_parts.append(text[match.start():line_end_index])
            index = line_end_index
            continue

        inner_content = text[brace_open_index + 1:brace_close_index]
        # Check if content is effectively empty (whitespace, nested empty braces, etc.)
        stripped = inner_content.strip()
        # Remove nested empty braces recursively
        while '{}' in stripped:
            stripped = stripped.replace('{}', '')
        stripped = stripped.strip()
        
        if stripped == "":
            # Replace with `foo = {}\n`, skipping possible trailing newline.
            output_parts.append(f"{assignment_prefix}{{}}\n")
            next_index = brace_close_index + 1
            if next_index < length and text[next_index] == "\n":
                next_index += 1
            index = next_index
        else:
            output_parts.append(text[match.start():brace_close_index + 1])
            index = brace_close_index + 1

    return "".join(output_parts)


RE_SPACER_ONLY_SUBVOICE = re.compile(
    r"(?sx)"
    r"(\\\\\{)"                        # subvoice start
    r"\s*(?:s[0-9.']*(?:\s+|$))+"       # one or more spacer durations
    r"\s*(\})"                          # closing brace
)


def _prune_spacer_only_subvoices(text: str) -> str:
    """
    Remove subvoices that contain only spacer rests (s1, s2., etc.).
    Repeats until no more matches are found.
    """
    previous = None
    current = text

    while previous != current:
        previous = current
        current = RE_SPACER_ONLY_SUBVOICE.sub("", current)

    return current


def _remove_spacer_notes(text: str) -> Tuple[str, int]:
    """
    Remove standalone spacer notes (s1*, s2*, s4*, s8*, etc.) that are layout placeholders.
    These are not real music content and add noise to training data.
    
    Note: Spacer notes in `forma` blocks (like `s1*59` for timing) are preserved
    as they serve a musical purpose for tempo/metrical structure.
    
    Pattern matches: s1*59, s2*12, s4*8, s8*4, etc. (but not in forma blocks)
    Also matches: s1, s2, s4, s8 (without multiplier) when standalone.
    
    Returns (cleaned_text, removed_count).
    """
    removed = 0
    
    # Protect spacer notes inside forma blocks (they're used for timing)
    # Find forma blocks and temporarily replace them with placeholders
    from lilynorm.stages.preprocessing.engrave_strip import _grab_balanced
    
    protected_blocks = {}
    block_counter = 0
    
    # Find all forma blocks
    forma_start_pattern = re.compile(r'forma\s*=\s*\{')
    index = 0
    
    while index < len(text):
        match = forma_start_pattern.search(text, index)
        if not match:
            break
        
        brace_start = match.end() - 1
        brace_end = _grab_balanced(text, brace_start, "{", "}")
        
        if brace_end != -1:
            block_key = f"__PROTECTED_FORMA_{block_counter}__"
            protected_blocks[block_key] = text[match.start():brace_end + 1]
            text = text[:match.start()] + block_key + text[brace_end + 1:]
            block_counter += 1
            index = match.start() + len(block_key)
        else:
            index = match.end()
    
    # Remove spacer notes with multipliers: s1*59, s2*12, etc. (outside forma blocks)
    pattern1 = re.compile(r'\bs\d+[*]\d+\b')
    text, count1 = pattern1.subn('', text)
    removed += count1
    
    # Remove standalone spacer notes on their own lines: s1, s2, s4, s8, etc.
    # But be careful not to remove them if they're part of actual music content
    pattern2 = re.compile(r'(?m)^\s*\bs\d+[.\']*\s*$')
    text, count2 = pattern2.subn('', text)
    removed += count2
    
    # Remove spacer notes followed by newline or end of line (but preserve context)
    # Pattern: "s8\n" or "s4 " at end of line
    pattern3 = re.compile(r'\bs\d+[.\']*\s+(?=\n|$)')
    text, count3 = pattern3.subn('', text)
    removed += count3
    
    # Restore protected forma blocks
    for block_key, block_content in protected_blocks.items():
        text = text.replace(block_key, block_content)
    
    return text, removed


def _compact_whitespace_aggressive(text: str) -> str:
    """
    Aggressively compact whitespace to reduce empty lines from ~25% to <10%.
    
    Rules:
    - Remove more than 1 consecutive empty line (keep max 1)
    - Preserve structure: keep 1 empty line between variable definitions and after closing braces
    - Remove empty lines before closing braces
    - Remove trailing empty lines
    """
    lines = text.split('\n')
    output = []
    prev_empty = False
    prev_was_var_def = False
    prev_was_closing_brace = False
    
    for i, line in enumerate(lines):
        is_empty = not line.strip()
        is_var_def = bool(re.match(r'^[A-Za-z_][\w-]*\s*=\s*\{', line))
        is_closing_brace = line.strip() == '}'
        
        if is_empty:
            # Only add empty line if:
            # 1. Previous line wasn't empty (max 1 consecutive empty line)
            # 2. Previous line was a variable definition or closing brace (preserve structure)
            if not prev_empty and (prev_was_var_def or prev_was_closing_brace):
                output.append('')
            prev_empty = True
            prev_was_var_def = False
            prev_was_closing_brace = False
        else:
            output.append(line)
            prev_empty = False
            prev_was_var_def = is_var_def
            prev_was_closing_brace = is_closing_brace
    
    # Remove trailing empty lines
    while output and not output[-1].strip():
        output.pop()
    
    return '\n'.join(output)


def _compact_spaces_safe(text: str) -> str:
    """
    Compact whitespace conservatively, preserving line structure and avoiding
    overly aggressive changes that might make diffs hard to read.
    """
    lines = text.splitlines()
    compacted = "\n".join(HSPACE.sub(" ", line).strip() for line in lines)

    compacted = RE_SPACE_BEFORE_CLOSER.sub("", compacted)
    compacted = RE_SPACE_AFTER_OPENER.sub("", compacted)
    compacted = RE_SPACE_BEFORE_PUNCT.sub("", compacted)
    compacted = RE_NOTE_OCTAVE_SPACE.sub("", compacted)
    compacted = RE_MULTI_BLANKS.sub("\n\n", compacted).strip() + "\n"

    return compacted


def _compact_spaces_simple(text: str) -> str:
    """
    Compact whitespace in a simpler way:
    - Replace 2+ spaces/tabs with a single space
    - Strip trailing spaces
    - Preserve line structure
    """
    lines = text.splitlines()
    normalized_lines = [
        re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in lines
    ]
    return "\n".join(normalized_lines).strip() + "\n"


# Categories used by StripOptions
CATEGORIES = ("overrides", "markups", "marks", "dynamics", "hairpins", "quotes")


# ---------------------------------------------------------------------------
# Options and configuration
# ---------------------------------------------------------------------------

@dataclass
class StripOptions:
    """
    Configuration for which engraving-related constructs to remove from
    LilyPond source, and which spacing mode to apply.
    """
    remove_overrides: bool = True
    remove_markups: bool = True
    remove_marks: bool = True
    remove_dynamics: bool = True
    remove_hairpins: bool = True
    remove_quotes: bool = True
    space_mode: str = DEFAULT_SPACE_MODE  # "safe" or "simple"

    @classmethod
    def from_sets(
        cls,
        remove: Iterable[str],
        keep: Iterable[str],
        *,
        space_mode: str = DEFAULT_SPACE_MODE,
    ) -> StripOptions:
        """
        Construct StripOptions from category names in `remove` and `keep`.

        NOTE: As implemented, categories default to True, and entries in
        `keep` are switched off. The `remove` set does not alter this default
        behavior; it is effectively redundant but preserved to avoid
        changing semantics.
        """
        flags = {name: True for name in CATEGORIES}

        for name in remove:
            if name in flags:
                flags[name] = True  # intentionally left as-is (no-op)

        for name in keep:
            if name in flags:
                flags[name] = False

        return cls(
            remove_overrides=flags["overrides"],
            remove_markups=flags["markups"],
            remove_marks=flags["marks"],
            remove_dynamics=flags["dynamics"],
            remove_hairpins=flags["hairpins"],
            remove_quotes=flags["quotes"],
            space_mode=space_mode,
        )


# ---------------------------------------------------------------------------
# Markup skipping and keyword removal
# ---------------------------------------------------------------------------

def _skip_markup_expression(source: str, index: int) -> int:
    """
    Starting from `index`, skip over a single markup expression.

    A markup expression is ONE of:
    - A quoted string "..."
    - A braced block {...}
    - A command like \\huge
    
    IMPORTANT: We only consume ONE element, not a sequence of them.
    This prevents consuming entire score blocks when processing markup
    like \\markup\\huge "title" that is followed by \\score.

    Returns the index immediately after the first markup element (or len(source)
    if it hits the end).
    """
    position = index
    length = len(source)

    # Skip leading whitespace
    while position < length and source[position] in " \t\r\n":
        position += 1

    if position >= length:
        return position

    char = source[position]

    # Block with braces: {...}
    if char == "{":
        end_index = _grab_balanced(source, position, "{", "}")
        return end_index + 1 if end_index != -1 else length

    # Quoted string: "..."
    if char == '"':
        position += 1
        escaped = False
        while position < length:
            current_char = source[position]
            position += 1
            if current_char == '"' and not escaped:
                return position
            escaped = (current_char == "\\") and not escaped
        return position

    # Command: \\something (consume just the command keyword, not what follows)
    if char == "\\":
        position += 1
        while (
            position < length
            and (source[position].isalnum() or source[position] in "_-")
        ):
            position += 1
        return position

    # Scheme: #... (consume to end of word/expression)
    if char == "#":
        position += 1
        while position < length and not source[position].isspace():
            position += 1
        return position

    # Fallback: plain word or token
    token_start = position
    while (
        position < length
        and not source[position].isspace()
        and source[position] not in '{}"'
    ):
        position += 1

    return position if position > token_start else token_start + 1


def _remove_common_macros(text: str) -> Tuple[str, int]:
    r"""Remove commonly seen macro definitions that are definitely not music.
    
    Only removes lines matching specific known macro patterns:
    - tr = \trill
    - su/giu = \change Staff = ...
    - dolce/arco/pizz/solo/soli/tu = markup
    - pad/padall = override
    - terzine/sestine = \tupletSpan
    - puntopz/fermopz/segnopz = \parenthesize
    - mbreak = { }
    
    Returns (cleaned_text, removed_count).
    """
    removed_count = 0
    
    # Remove lines with these exact macro names (case-sensitive)
    common_macros = (
        "tr", "su", "giu", "tremb",
        "dolce", "ten", "arco", "noarco", "pizz", 
        "soli", "solo", "tu", "tasto",
        "pad", "padall",
        "puntopz", "fermopz", "segnopz",
        "terzine", "terzinequarto", "sestine", "sestinequarto",
        "notypeset", "typeset", "senza", "con",
        "mbreak", "upl", "pratu"
    )
    
    lines = text.splitlines()
    filtered_lines = []
    
    for line in lines:
        # Check if line is an assignment to a known macro
        assign_match = re.match(r"^\s*([A-Za-z_][\w-]*)\s*=", line)
        if assign_match:
            var_name = assign_match.group(1)
            if var_name in common_macros:
                removed_count += 1
                continue  # Skip this line
        
        filtered_lines.append(line)
    
    return "\n".join(filtered_lines), removed_count


def _remove_macro_escape_usages(text: str) -> Tuple[str, int]:
    """Remove usages of known custom macro escapes left dangling in training mode.

    This targets inline tokens like "\\tr" and "\\solo" that are not LilyPond
    built-ins in this dataset and cause compilation errors if definitions are stripped.

    Returns (cleaned_text, removed_count).
    """
    # Keep the list conservative to avoid touching real LilyPond commands
    macro_escapes = (
        # performance and ornaments
        "tr", "solo", "soli", "tu",
        # dataset-specific helpers
        "upl", "pratu", "pad", "padall",
        "terzine", "terzinequarto", "sestine", "sestinequarto",
        "dolce", "tremb",
        # on/off markers frequently defined in variabili
        "con", "senza",
    )

    pattern = re.compile(r"\\(?:" + "|".join(sorted(set(macro_escapes), key=len, reverse=True)) + r")\b")
    updated, removed = pattern.subn("", text)
    return updated, removed


def _remove_markup_assignments(text: str) -> Tuple[str, int]:
    """Remove variable assignments that are pure markup directives.

    Example patterns:
        - `name = \\markup ...`
        - `name = _\\markup ...`
        - `name = ^\\markup { ... }`

    Returns (cleaned_text, removed_count).
    """
    removed = 0
    output_parts: List[str] = []
    search_start = 0

    pattern = re.compile(
        r"(?m)^[ \t]*[A-Za-z_][\w-]*\s*=\s*(?:[-_^]\s*)?\\markup\b"
    )

    while True:
        match = pattern.search(text, search_start)
        if not match:
            output_parts.append(text[search_start:])
            break

        output_parts.append(text[search_start:match.start()])
        after_markup = match.end()

        end_index = _skip_markup_expression(text, after_markup)
        while end_index < len(text) and text[end_index] in " \t\r":
            end_index += 1
        if end_index < len(text) and text[end_index] == "\n":
            end_index += 1

        removed += 1
        search_start = end_index

    cleaned = "".join(output_parts)

    # Also remove assignments to simple mark tokens like \mark "..." when used as pure assignment
    pattern_mark = re.compile(r"(?m)^\s*[A-Za-z_][\w-]*\s*=\s*_?\\mark\b.*$")
    cleaned, n2 = pattern_mark.subn("", cleaned)
    removed += n2

    # Squash extra blank lines from removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed


def _remove_instrument_setters(text: str) -> Tuple[str, int]:
    """Remove instrumentName/midiInstrument setters inside staff/voice blocks.
    
    Handles cases like:
      \\set Staff.instrumentName = \\markup\\center-column {"text"}
      \\set Staff.midiInstrument = #"string"
    """
    removed = 0
    
    # Remove midiInstrument first (simpler: just #"string")
    pattern = r'\\set\s+Staff\.midiInstrument\s*=\s*#"[^"]*"'
    count = len(re.findall(pattern, text))
    text = re.sub(pattern, "", text)
    removed += count
    
    # Remove instrumentName (more complex: has markup with nested braces)
    # Use iterative approach to handle the markup expression
    while True:
        match = re.search(r'\\set\s+Staff\.instrumentName\s*=\s*', text)
        if not match:
            break
        
        start = match.start()
        pos = match.end()
        
        # Skip whitespace
        while pos < len(text) and text[pos] in ' \t\n\r':
            pos += 1
        
        # Check for \markup
        if text[pos:pos+7] == '\\markup':
            pos += 7
            # Skip whitespace
            while pos < len(text) and text[pos] in ' \t\n\r':
                pos += 1
        
        # Now handle the markup commands and braces
        # Skip commands like \center-column, \huge, etc.
        while pos < len(text) and text[pos] == '\\':
            # Skip command
            pos += 1
            while pos < len(text) and (text[pos].isalnum() or text[pos] in '-_'):
                pos += 1
            # Skip whitespace
            while pos < len(text) and text[pos] in ' \t\n\r':
                pos += 1
        
        # Now handle the brace block or quoted string
        if pos < len(text) and text[pos] == '{':
            close_pos = _grab_balanced(text, pos, '{', '}')
            if close_pos != -1:
                pos = close_pos + 1
        elif pos < len(text) and text[pos] == '"':
            # Skip quoted string
            pos += 1
            while pos < len(text) and text[pos] != '"':
                if text[pos] == '\\':
                    pos += 2
                else:
                    pos += 1
            if pos < len(text):
                pos += 1
        
        # Remove from start to pos
        text = text[:start] + text[pos:]
        removed += 1
    
    # Clean up multiple spaces left behind by removals
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def _remove_empty_variable_assignments(text: str) -> Tuple[str, int]:
    """Remove variable assignments that are empty (leftover from content removal).
    
    Handles both simple empty blocks (varname = { }) and nested empty blocks
    (varname = { {} }).
    
    Also removes all references to those empty variables to avoid undefined references.
    
    Returns (cleaned_text, removed_count).
    """
    removed = 0
    
    def _is_empty_content(content: str) -> bool:
        """Check if content is effectively empty (only whitespace, nested empty braces, or clef directives)."""
        # Remove clef directives first (before whitespace removal, since they contain spaces)
        stripped = re.sub(r'\\clef\s+\w+', '', content)
        # Remove all whitespace and newlines
        stripped = re.sub(r'[\s\n]+', '', stripped)
        # Remove all nested empty braces {} recursively
        while '{}' in stripped:
            stripped = stripped.replace('{}', '')
        # If nothing remains, it's empty
        return len(stripped.strip()) == 0
    
    # First pass: find all empty variable assignments and their positions
    empty_var_ranges = []  # List of (start_pos, end_pos, var_name)
    pattern = re.compile(r"(?m)^([A-Za-z_][\w-]*)\s*=\s*\{")
    index = 0
    
    while index < len(text):
        match = pattern.search(text, index)
        if not match:
            break
        
        var_name = match.group(1)
        assignment_start = match.start()
        brace_start = match.end() - 1  # Position of opening brace
        brace_end = _grab_balanced(text, brace_start, "{", "}")
        
        if brace_end != -1:
            # Extract content between braces
            content = text[brace_start + 1:brace_end]
            
            # Check if content is effectively empty
            if _is_empty_content(content):
                # Find the end of the assignment (including all trailing newlines)
                assignment_end = brace_end + 1
                # Include all trailing newlines
                while assignment_end < len(text) and text[assignment_end] == '\n':
                    assignment_end += 1
                empty_var_ranges.append((assignment_start, assignment_end, var_name))
                removed += 1
            
            index = brace_end + 1
        else:
            index = match.end()
    
    # Second pass: remove empty variable definitions (in reverse order to preserve indices)
    for start, end, var_name in reversed(empty_var_ranges):
        text = text[:start] + text[end:]
        # Remove all references to this variable (e.g., \Ivl)
        ref_pattern = re.compile(rf"\\{re.escape(var_name)}\b")
        text = ref_pattern.sub("", text)
    
    # Match variable assignment with no value (just whitespace after =)
    pattern2 = re.compile(r"(?m)^[A-Za-z_][\w-]*\s*=\s*$")
    text, count2 = pattern2.subn("", text)
    removed += count2
    
    # Clean up extra blank lines from removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def _light_cleanup(text: str) -> str:
    """Minimal safety fixes used when engravings are kept.

    Avoids structural removals (lyrics, custom macros, empty blocks, spacer pruning),
    and only fixes syntax breakers we've seen in this dataset.
    """
    # Remove unsupported custom commands (but NOT variable references!)
    # These are standalone commands that break compilation
    # NOTE: 'forma' is NOT in this list because it contains \key and \time which are ESSENTIAL
    unsupported = (
        "mbreak",
        "trasp",
        "notrasp",
        "typeset",
        "notypeset",
        # "forma",  # REMOVED: contains \key and \time (key signatures) - essential for music
        "terzine",
        "con",
        "senza",
        "fermopz",
        "terzinesenza",
        "terzinecon",
        "pizz",
        "arco",
        "fort",
        "staccatissimo",
        "tasto",
        # Note: "tu" removed from list - handled separately below due to whitespace issues
    )
    
    # Handle \tu separately - but NOT when it's part of \tuplet!
    # Use negative lookahead to ensure we don't match \tuplet
    text = re.sub(r"\\tu(?!plet)\s*", "", text)
    
    text = re.sub(r"\\(?:" + "|".join(unsupported) + r")\b", "", text)
    
    # Fix scheme errors: ## { # } -> remove these malformed empty scheme blocks
    text = re.sub(r"##\s*\{\s*#\s*\}", "", text)
    
    # DO NOT remove roman numeral variables like \Ibcn, \IIbfn, etc.
    # These are movement-specific music/figuremode assignments that hold actual content.

    # Figured bass broken across lines: "<\n  ->" -> "<->"
    text = re.sub(r"<\s*\n\s*([+\-\d\s]+>)", r"<\1", text)

    # Dotted duration inheritance issues
    text = re.sub(r"(\d+)\.(\([a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s*\))", r"\1\2\3\4\5", text)
    text = re.sub(r"(\d+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.", r"\1\2\3\4", text)

    # Missing dot in revert paths: \revert Stem #'transparent -> \revert Stem.#'transparent
    text = re.sub(r"(\\revert\s+\w+)\s+#'", r"\1.#'", text)

    # Strip stray \once that are not followed by a Lily command (breaks timed music)
    text = RE_LONE_ONCE.sub("", text)

    # Normalize legacy trill macro usages to the built-in command
    text = re.sub(r"\\tr\b", r"\\trill", text)
    text = re.sub(r"\\tr(?=[\\_])", r"\\trill", text)

    # Remove lyric-style extenders that accidentally cling to note tokens (e.g., si_)
    text = re.sub(r"\b((?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d?)[_]+\b", r"\1", text)
    # More aggressive underscore stripping (covers cases next to punctuation/newlines)
    text = re.sub(r"((?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d?)[_]+", r"\1", text)
    # Remove standalone lyric underscores left as separate tokens
    text = re.sub(r"\s+_\s+", " ", text)
    
    # Fix invalid note name 'x' at end of line (percussion/cross-staff notation)
    # Pattern: "sold,2x" -> "sold,2 do"
    text = re.sub(r"(\b[a-z]+[',]*,?)2\s*x\b", r"\1 2 do", text)
    
    # Fix figured bass broken syntax: '+[' and variants
    text = re.sub(r"\s*\+\[", " \\\\[ ", text)
    text = re.sub(r"\s*\+\]", " \\\\] ", text)
    
    # Remove malformed variable assignments with -align, empty markup, or ^{ patterns
    text = re.sub(r'^[a-z]+\s*=\s*\^-align\s+"[^"]*"\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[a-z]+\s*=\s*\^\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[a-z]+\s*=\s*\^\s*\{[^}]*\}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[a-z]+\s*=\s*\^\{[^}]*\}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove underscore tokens that directly precede a note without trailing space (e.g., _mi)
    text = re.sub(r"\s*_(?=(?:do|re|mi|fa|sol|la|si|[a-gr]))", " ", text)
    
    # Fix note names with sharp/natural/flat modifiers: "mi!16" -> "mi16", "fa#2" -> "fa2"
    # These are malformed - the modifier should be after duration or part of syntax
    text = re.sub(r"((?:do|re|mi|fa|sol|la|si)[',]*?)([!#b]+)(\d+)", r"\1\3", text)
    
    # Fix orphaned "r" rest on its own line appearing as note name
    # Pattern: lines that are just "r" or "r8" or "r4" when they should be rests within music
    text = re.sub(r"^\s*r(?:8|4|2|1)?\s*$", "", text, flags=re.MULTILINE)
    
    # Fix \tempo with malformed arguments: "\tempo 2. = 60" (should be "\tempo 2.=60" or removed)
    # Remove tempo marks that have syntax issues
    text = re.sub(r"\\\\tempo\s+[^=]+=\s*\d+", "", text)
    # Final fallback: strip any remaining bare underscores
    text = re.sub(r"_+", " ", text)

    # Remove stray + characters that cling to notes (e.g., la+)
    text = re.sub(r"((?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d?)\+", r"\1", text)
    # Remove leading + tokens at line starts before notes
    text = re.sub(r"(?m)^[ \t]*\+(?=\s*[a-gr])", "", text)
    # Collapse standalone + or - tokens between whitespace
    text = re.sub(r"\s+[\+-](?=\s)", " ", text)
    
    # Fix repeating note patterns: Note^ followed by same/different note (NOTENAME_PITCH error)
    # Pattern: "mi^ la la" -> "mi la la" (remove the ^ articulation that confuses the parser)
    text = re.sub(r"(\b(?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d?)\^\s+(?=(?:do|re|mi|fa|sol|la|si|[a-gr]))", r"\1 ", text)
    
    # Remove \vspace from middle of markup strings
    text = re.sub(r'"\\\\vspace[^"]*', r'"', text)
    
    # Note: \tasto, \fort, \staccatissimo now in unsupported list above
    
    # Fix figured bass figure alteration errors: < !> -> remove the !>
    # Pattern: "< !>2." becomes "<>2."
    text = re.sub(r"<\s*!\s*>", "<>", text)
    
    # Remove problematic one-liners: variable = articulation (^ or ^-align or ^\markup)
    # These break compilation: "presto = ^ ", "ts = ^", etc.
    text = re.sub(r'^[a-z]+\s*=\s*\^(?:-align\s+"[^"]*"|\\\\markup)?\s*(?:\{[^}]*\})?\s*$', '', text, flags=re.MULTILINE)

    # Neutralize undefined wrapper blocks like <<\IIvlIn\forma>> or <<\IIbcn\forma\IIbfn>>
    # These are references to undefined variables that break compilation
    # Use DOTALL to match across newlines, and allow zero or more spaces between elements
    # Match 1-3 backslash identifiers
    text = re.sub(r"<<\s*(?:\\[A-Za-z0-9_]+\s*){1,3}>>", "{}", text, flags=re.DOTALL)

    # Drop inline markup tokens that appear inside music lines (e.g., \markup {\musicglyph ...})
    # These are visual-only and can break parsing when left between notes.
    # Handle nested braces by removing markup recursively
    while re.search(r"\\markup\s*\{[^{}]*\}", text):
        text = re.sub(r"\\markup\s*\{[^{}]*\}", "", text)
    # Also remove simple inline markups without braces, e.g., \markup\italic"Tasto Solo"
    text = re.sub(r"\\markup(?:\s*\\[A-Za-z]+)*\s*\"[^\"]*\"", "", text)
    text = re.sub(r"\\markup(?:\s*\\[A-Za-z]+)+", "", text)
    # Remove text alignment directives like ^-align {\musicglyph ...}
    text = re.sub(r"[\^_-]*align\s*\\[A-Za-z]+\s*#\"[^\"]*\"", "", text)
    text = re.sub(r"[\^_-]*align\s*\{[^{}]*\}", "", text)
    
    # Remove malformed variable assignments where the variable name is a string (e.g., "|" = \bar "||")
    # These break parsing completely
    text = re.sub(r"^\s*\"[^\"]*\"\s*=.*$", "", text, flags=re.MULTILINE)

    # Clean malformed figured-bass brackets with stray +/- tokens (e.g., < +>, < - 6>)
    text = re.sub(r"<\s*[\+-]\s*>", "<>", text)
    text = re.sub(r"<\s*[\+-]\s+(\d)", r"<\1", text)
    text = re.sub(r"(\d)\s+[\+-]\s*>", r"\1>", text)
    text = re.sub(r"(\d)\s+[\+-]\s+(\d)", r"\1 \2", text)

    # Ensure slur parentheses have spaces so tokens parse (mi( sol) -> mi( sol ) )
    text = re.sub(r"\((?=[a-grA-G])", "( ", text)
    text = re.sub(r"(?<=[a-gr0-9',])\)", " )", text)

    return text


def _remove_standalone_markup_lines(text: str) -> Tuple[str, int]:
    """Remove non-musical, standalone markup/directive lines.

    Targets lines that begin with layout-only commands, not embedded in music:
    - \\markup ..., \\halign, \\center-column, \\musicglyph, \\vspace
    - \\pageBreak, \\pointAndClickOff
    Note: Do NOT remove \\language; pitch names (do, re, mi) depend on it.

    Returns (cleaned_text, removed_count).
    """
    patterns = [
        r"(?m)^\s*\\markup.*$",
        r"(?m)^\s*\\halign\b.*$",
        r"(?m)^\s*\\center-column\b.*$",
        r"(?m)^\s*\\musicglyph\b.*$",
        r"(?m)^\s*\\vspace\b.*$",
        r"(?m)^\s*\\pageBreak\s*$",
        r"(?m)^\s*\\pointAndClickOff\s*$",
    ]
    removed = 0
    for pat in patterns:
        before = text
        text = re.sub(pat, "", text)
        if text != before:
            removed += 1
    # Collapse excessive blank lines introduced by removals
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def _remove_standalone_quoted_lines(text: str) -> Tuple[str, int]:
    """Remove lines that are just quoted strings (often leftover from markup stripping)."""
    pattern = re.compile(r'(?m)^\s*"[^\n"]*"\s*$')
    cleaned, removed = pattern.subn("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed


def _remove_empty_block_directives(text: str, directives: Tuple[str, ...]) -> Tuple[str, int]:
    """Remove empty \\directive { } blocks (whitespace only inside)."""
    removed = 0
    for directive in directives:
        pattern = re.compile(rf"(?s)\\{directive}\s*\{{\s*\}}")
        text, n = pattern.subn("", text)
        removed += n
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def _final_cleanup(text: str) -> str:
    """Dataset-tail cleanup that used to live in the driver.

    Runs after engraving stripping (or directly if engravings are kept).
    """
    # Remove unsupported custom commands and movement-local roman macros
    # NOTE: 'forma' is NOT in this list because it contains \key and \time which are ESSENTIAL
    unsupported = (
        "mbreak",
        "trasp",
        "notrasp",
        "typeset",
        "notypeset",
        "Voice",
        # "forma",  # REMOVED: contains \key and \time (key signatures) - essential for music
        "terzine",
        "con",
        "senza",
        "notrasp",
    )
    text = re.sub(r"\\(?:" + "|".join(unsupported) + r")\b", "", text)
    text = re.sub(r"\\(?:I|II|III|IV)[A-Za-z][\w-]*\b", "", text)

    # Structural cleanups - DISABLED: these patterns are too aggressive and destroy valid structures
    # The [^>]* pattern is greedy and matches across actual music content
    # Leaving only safe empty-block removal
    text = re.sub(r"<<\s*>>", "", text)
    # DISABLED: text = re.sub(r"\\new\s+(?:Staff|ChoirStaff|StaffGroup)\s*<<[^>]*>>", "", text)
    # DISABLED: text = re.sub(r"(?m)^\\new\s+(?:Voice|Lyrics)\s*=\s*\"[^\"]*\"\s*$", "", text)
    text = re.sub(r"^[A-Za-z_][\w-]*\s*=\s*\{\s*\}", "", text, flags=re.MULTILINE)
    # REMOVED: This pattern was removing variables with \key and \time which are MUSICAL STRUCTURE, not engraving!
    # text = re.sub(r"(?ms)^[A-Za-z_][\w-]*\s*=\s*\{\s*(?:\\(?:clef|key|time)\s+[^\}]*\s*)*\}\s*$", "", text)
    text = re.sub(r"(?m)^\s*>>\s*$", "", text)
    text = re.sub(r"(?m)(#\([^)]*\).*\n)\n*\}\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)(\\pageBreak\s*\n)\n*\}\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\}\s*\n\s*\}\s*$", "}", text, flags=re.MULTILINE)
    text = re.sub(r"(?s)\\score\s*\{\s*>>\s*(?:\\(?:midi|layout)\s*\{[^}]*\}\s*)*\}", "", text)

    # Token tweaks
    text = re.sub(r"(?m)^\s*\+\s*", "", text)
    text = re.sub(r"(?<=\s)\+(?=\s)", "", text)
    text = re.sub(r"(?<=\w)\+(?=[)\]\}])", "", text)
    text = re.sub(r"(?<=\.)\+", "", text)
    text = re.sub(r"(?<=\w)\+(?=\s|$)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^\s+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*(?:up|down)\s*$", "", text)
    text = re.sub(r"(?m)^([A-Za-z_][\w-]*)\s*=\s*(?:=\s+|$)", r"\1 = {}", text)
    text = re.sub(r"(?m)^\s*=\s+\w+\s*$", "", text)

    # Musical dotted patterns / figured bass / revert fixes
    text = re.sub(r"(\d+)\.(\([a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s*\))", r"\1\2\3\4\5", text)
    text = re.sub(r"(\d+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.", r"\1\2\3\4", text)
    text = re.sub(r"<\s*\n\s*([+\-\d\s]+>)", r"<\1", text)
    text = re.sub(r"(\\revert\s+\w+)\s+#'", r"\1.#'", text)

    # Version line fix at top
    if text.startswith('version "'):
        quote_end = text.find('"', 9)
        if quote_end > 9:
            version_line = text[:quote_end + 1]
            if not version_line.rstrip().endswith('"'):
                text = version_line.rstrip() + '"' + text[quote_end:]

    return text


def _eat_after_keyword(
    source: str,
    keyword_regex: re.Pattern,
    *,
    deep_markup: bool = False,
) -> Tuple[str, int]:
    """
    Remove content following a keyword pattern (e.g. \\markup, \\mark).

    Behavior:
      - Search for `keyword_regex` matches in `source`.
      - After each match, skip surrounding spaces and:
        * If deep_markup=True, call _skip_markup_expression().
        * If next char is '{', remove a balanced block.
        * If next char is '"', remove a quoted string.
        * Otherwise remove a single token, possibly followed by a quoted string.

    Returns (new_source, removed_count).
    """
    removed_count = 0
    search_start = 0
    output_parts: List[str] = []

    while True:
        match = keyword_regex.search(source, search_start)
        if not match:
            output_parts.append(source[search_start:])
            break

        output_parts.append(source[search_start:match.start()])
        after_keyword_index = match.end()

        # Skip immediate horizontal whitespace
        while (
            after_keyword_index < len(source)
            and source[after_keyword_index] in " \t"
        ):
            after_keyword_index += 1

        # Deep markup mode: delegate to _skip_markup_expression
        if deep_markup:
            search_start = _skip_markup_expression(source, after_keyword_index)
            removed_count += 1
            continue

        # Block { ... }
        if after_keyword_index < len(source) and source[after_keyword_index] == "{":
            end_index = _grab_balanced(source, after_keyword_index, "{", "}")
            if end_index != -1:
                search_start = end_index + 1
                removed_count += 1
                continue

        # Quoted string "..."
        if after_keyword_index < len(source) and source[after_keyword_index] == '"':
            quote_index = after_keyword_index + 1
            while quote_index < len(source) and source[quote_index] != '"':
                if source[quote_index] == "\\" and quote_index + 1 < len(source):
                    quote_index += 2
                else:
                    quote_index += 1
            search_start = (quote_index + 1) if quote_index < len(source) else len(source)
            removed_count += 1
            continue

        # Single token
        token_end_index = after_keyword_index
        while (
            token_end_index < len(source)
            and not source[token_end_index].isspace()
            and source[token_end_index] not in '{}"'
        ):
            token_end_index += 1

        # Optional quoted string directly after the token
        scan_index = token_end_index
        while scan_index < len(source) and source[scan_index] in " \t":
            scan_index += 1

        if scan_index < len(source) and source[scan_index] == '"':
            quote_index = scan_index + 1
            while quote_index < len(source) and source[quote_index] != '"':
                if source[quote_index] == "\\" and quote_index + 1 < len(source):
                    quote_index += 2
                else:
                    quote_index += 1
            token_end_index = (quote_index + 1) if quote_index < len(source) else len(source)

        search_start = token_end_index
        removed_count += 1

    return "".join(output_parts), removed_count


# ---------------------------------------------------------------------------
# Main stripping logic
# ---------------------------------------------------------------------------

def _strip_inline_patterns(
    text: str,
    options: StripOptions,
) -> Tuple[str, Dict[str, int]]:
    """
    Strip engraving-related patterns (overrides, dynamics, markups, etc.) from
    LilyPond source according to StripOptions.

    Returns (cleaned_text, removal_counts).
    """
    counts: Dict[str, int] = {
        "overrides": 0,
        "markups": 0,
        "marks": 0,
        "dynamics": 0,
        "hairpins": 0,
        "quotes": 0,
        "metadata": 0,
        "custom_assignments": 0,
    }

    # Attached quotes like ^"text"
    if options.remove_quotes:
        text, removed_quotes = RE_ATTACHED_QUOTES.subn("", text)
        text, removed_inline = RE_INLINE_QUOTES.subn("", text)
        counts["quotes"] += removed_quotes + removed_inline

    # Header-like metadata
    text, removed_headers = _remove_metadata_headers(text)
    counts["metadata"] += removed_headers

    # \\markup and \\mark
    if options.remove_markups:
        text, removed_markups = _eat_after_keyword(
            text,
            RE_MARKUP,
            deep_markup=True,
        )
        counts["markups"] += removed_markups

    if options.remove_marks:
        text, removed_marks = _eat_after_keyword(text, RE_MARK)
        counts["marks"] += removed_marks
        
        # Also remove performance marks like \soli, \tu, etc.
        text, removed_perf = RE_PERFORMANCE_MARKS.subn("", text)
        counts["marks"] += removed_perf

    # Overrides and related engraving commands
    if options.remove_overrides:
        # IMPORTANT: Protect \layout, \midi, \paper, \header blocks from override removal
        # These blocks are critical for output generation and should not be modified
        protected_blocks = {}
        text_protected = text
        block_counter = 0
        
        # Extract all layout/midi/paper/header blocks
        for directive in ("layout", "midi", "paper", "header"):
            pattern = re.compile(rf"\\{directive}\s*\{{", re.M)
            search_start = 0
            
            while True:
                match = pattern.search(text_protected, search_start)
                if not match:
                    break
                
                brace_open_index = match.end() - 1
                brace_close_index = _grab_balanced(text_protected, brace_open_index, "{", "}")
                
                if brace_close_index == -1:
                    search_start = match.end()
                    continue
                
                block_key = f"__PROTECTED_BLOCK_{block_counter}__"
                protected_blocks[block_key] = text_protected[match.start():brace_close_index + 1]
                text_protected = text_protected[:match.start()] + block_key + text_protected[brace_close_index + 1:]
                block_counter += 1
                search_start = match.start() + len(block_key)
        
        text = text_protected
        
        # Remove \\with { ... } blocks
        updated_text, removed_with_blocks = _remove_with_blocks(text)
        if removed_with_blocks:
            text = updated_text
            counts["overrides"] += removed_with_blocks

        # Inline overrides/tweaks/shape/omit...
        for regex in RE_OVERRIDES:
            updated_text, removed = regex.subn("", text)
            if removed:
                text = updated_text
                counts["overrides"] += removed
        
        # Restore protected blocks
        for block_key, block_content in protected_blocks.items():
            text = text.replace(block_key, block_content)

    # Dynamics & hairpins
    if options.remove_dynamics:
        text, removed_dynamics = RE_DYNAMICS.subn("", text)
        counts["dynamics"] += removed_dynamics

    if options.remove_hairpins:
        text, removed_hairpins = RE_HAIRPINS.subn("", text)
        counts["hairpins"] += removed_hairpins

    # Remove dataset-specific custom commands and their variable assignments (excluding \forma)
    text, removed_custom = RE_CUSTOM_COMMANDS.subn("", text)
    counts["custom_assignments"] += removed_custom
    text, removed_custom_assigns = _remove_custom_assignments(text)
    counts["custom_assignments"] += removed_custom_assigns
    
    # Clean up stranded attachment markers and lone \\once
    # But first protect Scheme expressions like #(set-default-paper-size "a4")
    # from having their hyphens removed
    text, scheme_mapping = _protect_scheme_expressions(text)
    text, _ = RE_STRAY_ATTACH.subn("", text)
    text = _restore_scheme_expressions(text, scheme_mapping)
    text, _ = RE_LONE_ONCE.subn("", text)

    # Remove lyricmode content
    text, _ = _strip_lyricmode_assignments(text)
    text, _ = _strip_inline_lyricmode(text)

    # Normalize empty assignments and remove empty blocks/assignments
    text = _collapse_empty_assignment_blocks(text)
    
    # Remove empty angle brackets created by content removal
    text = re.sub(r"<<\s*>>", "", text)
    
    # DISABLED: Remove incomplete \new Voice/Lyrics declarations - pattern is too greedy
    # text = re.sub(r"(?m)^\\new\s+(?:Voice|Lyrics)\s*=\s*\"[^\"]*\"\s*$", "", text)
    
    # Remove incomplete \new Staff structures with no content
    text = re.sub(r"\\new\s+(?:Staff|ChoirStaff|StaffGroup)\s*<<[^>]*>>", "", text)
    
    # Remove stray '>>' lines
    text = re.sub(r"(?m)^\s*>>\s*$", "", text)
    
    # Now remove empty blocks and assignments after structure cleanup
    text = RE_EMPTY_BLOCK_LINE.sub("", text)
    text = RE_EMPTY_ASSIGNMENT_LINE.sub("", text)
    
    # Replace inline empty braces with space, but only if not part of an assignment
    text = RE_INLINE_EMPTY_BRACES.sub(" ", text)
    
    text = RE_INCLUDE_TAG.sub("", text)
    text = RE_REPEATED_INCLUDE.sub("", text)
    text = RE_EMPTY_SCORES.sub("", text)
    text = RE_EMPTY_LAYOUT_BLOCK.sub("", text)

    if DROP_EMPTY_ASSIGNMENTS:
        text = re.sub(r"(?m)^\s*\w+\s*=\s*\{\s*\}\s*$", "", text)

    if PRUNE_SPACER_SUBVOICES:
        text = _prune_spacer_only_subvoices(text)
    
    # Remove orphaned closing braces that follow Scheme commands or page breaks
    text = re.sub(r"(?m)(#\([^)]*\).*\n)\n*\}\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)(\\pageBreak\s*\n)\n*\}\s*$", r"\1", text, flags=re.MULTILINE)
    
    # Remove orphaned direction symbols that appear on their own lines (up, down)
    text = re.sub(r"(?m)^\s*(?:up|down)\s*$", "", text)
    
    # Remove broken assignments like "su = = up" (incomplete after macro removal)
    text = re.sub(r"(?m)^([A-Za-z_][\w-]*)\s*=\s*(?:=\s+|$)", r"\1 = {}", text)
    
    # Remove standalone "= something" lines that are broken remnants
    text = re.sub(r"(?m)^\s*=\s+\w+\s*$", "", text)
    
    # Fix dotted note patterns with inherited durations in slurs
    # Pattern: X8.(Y. Z. W.) -> X8(Y Z W)  (remove dots from notes that inherit duration)
    text = re.sub(r"(\d+)\.(\([a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s*\))", r"\1\2\3\4\5", text)
    
    # Fix dotted note runs with inherited durations
    # Pattern: X16. Y. Z. W. X8 -> X16 Y Z W X8 (add explicit durations or remove inheriting dots)
    text = re.sub(r"(\d+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.", r"\1\2\3\4", text)
    
    # Fix broken figured bass notation: < on one line, alteration/number on next
    # Pattern: < followed by newline and whitespace, then alteration/number and >
    # Example: "<\n       ->" should become "<->"
    text = re.sub(r"<\s*\n\s*([+\-\d\s]+>)", r"<\1", text)
    
    # Fix \revert commands with missing dot before property
    # Pattern: \revert Stem #'transparent -> \revert Stem.#'transparent
    text = re.sub(r"(\\revert\s+\w+)\s+#'", r"\1.#'", text)

    # Whitespace compaction
    if options.space_mode == "simple":
        text = _compact_spaces_simple(text)
    else:
        text = _compact_spaces_safe(text)

    return text, counts


def clean_lilypond(src: str, options: StripOptions) -> Tuple[str, Dict[str, int]]:
    """
    Entry point for cleaning LilyPond engraving/markup from a source string.

    Returns (cleaned_source, counts_dict).
    """
    text, counts = _strip_inline_patterns(src, options)
    return text, counts


# ---------------------------------------------------------------------------
# Integration with lilynorm NormOptions
# ---------------------------------------------------------------------------

try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        """
        Fallback NormOptions when lilynorm.utils.options is unavailable.

        Only provides the attribute `keep_engraving` used in this module.
        """
        keep_engraving: bool = True  # default: keep engravings


def run(text: str, opts: NormOptions) -> str:
    r"""
    Main entry point used by the lilynorm pipeline.

    If opts.keep_engraving is True, apply only light cleanup.
    If False, remove \paper blocks for ML training (step 1 of gradual noise removal).
    """
    if getattr(opts, "keep_engraving", True):
        print("[engrave_strip] keeping engravings", file=sys.stderr)
        cleaned = _light_cleanup(text)
        
        # Remove spacer notes (layout placeholders, not real music)
        cleaned, spacer_count = _remove_spacer_notes(cleaned)
        if spacer_count > 0:
            print(f"[engrave_strip] removed {spacer_count} spacer note(s)", file=sys.stderr)
        
        # Remove empty variable assignments even when keeping engravings
        cleaned, empty_var_count = _remove_empty_variable_assignments(cleaned)
        if empty_var_count > 0:
            print(f"[engrave_strip] removed {empty_var_count} empty variable assignment(s)", file=sys.stderr)
        
        # Aggressively compact whitespace to reduce noise
        cleaned = _compact_whitespace_aggressive(cleaned)
    else:
        print("[engrave_strip] removing paper, top-level scheme, and common macros", file=sys.stderr)
        # Step 1: Remove \paper blocks (safe - self-contained blocks)
        cleaned, paper_count = _remove_block_directive(text, "paper")
        if paper_count > 0:
            print(f"[engrave_strip] removed {paper_count} paper block(s)", file=sys.stderr)

        # Step 2: Remove \header blocks (metadata not needed for ML training)
        cleaned, header_count = _remove_block_directive(cleaned, "header")
        if header_count > 0:
            print(f"[engrave_strip] removed {header_count} header block(s)", file=sys.stderr)

        # Step 3: Remove top-level Scheme blocks (set-default-paper-size, set-global-staff-size, custom let)
        cleaned, scheme_count = _remove_top_level_scheme_blocks(cleaned)
        if scheme_count > 0:
            print(f"[engrave_strip] removed {scheme_count} scheme block(s)", file=sys.stderr)

        # Step 4: Remove common macro definitions (tr, dolce, pad, etc.)
        cleaned, macro_count = _remove_common_macros(cleaned)
        if macro_count > 0:
            print(f"[engrave_strip] removed {macro_count} macro definition(s)", file=sys.stderr)

        # Step 4b: Remove dangling usages of those macros (e.g., \tr, \solo)
        cleaned, use_count = _remove_macro_escape_usages(cleaned)
        if use_count > 0:
            print(f"[engrave_strip] removed {use_count} macro usage(s)", file=sys.stderr)

        # Step 4c: Remove markup-only variable assignments (e.g., ds = _\markup ...)
        cleaned, markup_assign_count = _remove_markup_assignments(cleaned)
        if markup_assign_count > 0:
            print(f"[engrave_strip] removed {markup_assign_count} markup assignment(s)", file=sys.stderr)

        # Step 4d: Remove dataset-specific helper commands and assignments (e.g., notrasp = ...)
        cleaned, removed_custom_cmds = RE_CUSTOM_COMMANDS.subn("", cleaned)
        cleaned, removed_custom_assigns = _remove_custom_assignments(cleaned)
        removed_custom = removed_custom_cmds + removed_custom_assigns
        if removed_custom:
            print(f"[engrave_strip] removed {removed_custom} custom helper definition(s)", file=sys.stderr)

        # Step 4e: Remove remaining inline markups (textual ornaments, directions)
        cleaned, inline_markup_count = _eat_after_keyword(cleaned, RE_MARKUP, deep_markup=True)
        if inline_markup_count:
            print(f"[engrave_strip] removed {inline_markup_count} inline markup token(s)", file=sys.stderr)

        # Step 4f: Remove attached/inlined quoted annotations like r8"sempre piano"
        cleaned, attached_quote_count = RE_ATTACHED_QUOTES.subn("", cleaned)
        cleaned, inline_quote_count = RE_INLINE_QUOTES.subn("", cleaned)
        total_quotes = attached_quote_count + inline_quote_count
        if total_quotes:
            print(f"[engrave_strip] removed {total_quotes} inline quoted annotation(s)", file=sys.stderr)

        # Step 4b: Remove instrument/midi setters in staff blocks
        cleaned, inst_count = _remove_instrument_setters(cleaned)
        if inst_count > 0:
            print(f"[engrave_strip] removed {inst_count} instrument/midi setter group(s)", file=sys.stderr)

        # Step 4: Remove standalone markup/directive lines
        cleaned, markup_count = _remove_standalone_markup_lines(cleaned)
        if markup_count > 0:
            print(f"[engrave_strip] pruned {markup_count} markup/directive groups", file=sys.stderr)

        cleaned, quote_line_count = _remove_standalone_quoted_lines(cleaned)
        if quote_line_count > 0:
            print(f"[engrave_strip] removed {quote_line_count} standalone quoted line(s)", file=sys.stderr)

        cleaned, empty_book_count = _remove_empty_block_directives(cleaned, ("bookpart", "book"))
        if empty_book_count > 0:
            print(f"[engrave_strip] removed {empty_book_count} empty book block(s)", file=sys.stderr)

        # Step 5-6: Remove \layout/\midi/\score blocks for pure ML training
        if STRIP_SCORE_LAYOUT:
            cleaned, layout_count = _remove_block_directive(cleaned, "layout")
            cleaned, midi_count = _remove_block_directive(cleaned, "midi")
            if layout_count or midi_count:
                print(f"[engrave_strip] removed layout={layout_count} midi={midi_count} block(s)", file=sys.stderr)

            cleaned, score_count = _remove_block_directive(cleaned, "score")
            if score_count:
                print(f"[engrave_strip] removed {score_count} score block(s)", file=sys.stderr)

            cleaned, empty_book_count = _remove_empty_block_directives(cleaned, ("bookpart", "book"))
            if empty_book_count > 0:
                print(f"[engrave_strip] removed {empty_book_count} empty book block(s)", file=sys.stderr)
        else:
            print("[engrave_strip] keeping score/layout/midi blocks (PDF intact)", file=sys.stderr)

        # Step 7: Remove inline engraving overrides/tweaks/shape/omit directives
        # Only when stripping layout blocks (pure ML training mode)
        # When keeping layout, preserve all overrides for proper PDF rendering
        if STRIP_SCORE_LAYOUT:
            overrides_removed_total = 0
            for regex in RE_OVERRIDES:
                updated_text, removed = regex.subn("", cleaned)
                if removed:
                    cleaned = updated_text
                    overrides_removed_total += removed
            if overrides_removed_total:
                print(f"[engrave_strip] removed {overrides_removed_total} inline override/tweak directives", file=sys.stderr)
        else:
            print("[engrave_strip] keeping override directives for layout preservation", file=sys.stderr)

        # Apply light cleanup for syntax fixes
        cleaned = _light_cleanup(cleaned)
        
        # Remove spacer notes (layout placeholders, not real music)
        cleaned, spacer_count = _remove_spacer_notes(cleaned)
        if spacer_count > 0:
            print(f"[engrave_strip] removed {spacer_count} spacer note(s)", file=sys.stderr)
        
        # Final step: Remove empty variable assignments (leftovers from content stripping)
        cleaned, empty_var_count = _remove_empty_variable_assignments(cleaned)
        if empty_var_count > 0:
            print(f"[engrave_strip] removed {empty_var_count} empty variable assignment(s)", file=sys.stderr)
        
        # Aggressively compact whitespace to reduce noise
        cleaned = _compact_whitespace_aggressive(cleaned)

    return cleaned
