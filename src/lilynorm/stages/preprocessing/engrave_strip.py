from __future__ import annotations

import re
import sys
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# Active stripping steps
# ---------------------------------------------------------------------------


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


def _grab_angles(text: str, start: int) -> int:
    """
    Given an index `start` pointing at a '<<' opener, return the index
    immediately after the matching '>>', handling nested blocks.

    Returns -1 if the text ends before a matching '>>' is found.
    """
    depth = 1
    index = start + 2
    length = len(text)

    while index < length and depth > 0:
        if text.startswith("<<", index):
            depth += 1
            index += 2
        elif text.startswith(">>", index):
            depth -= 1
            index += 2
        else:
            index += 1

    return index if depth == 0 else -1


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
    r"""Drop header-style metadata lines that add noise for training (keep \language for pitch names)."""
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
RE_FOOTNOTE = re.compile(r"(?:[-_^]\s*)?\\footnote\b", re.I)

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
RE_INLINE_QUOTES = re.compile(r'(?m)(?<=\\S)\\s*\"[^\"\\n]*\"')
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


# ---------------------------------------------------------------------------
# Markup skipping and keyword removal
# ---------------------------------------------------------------------------

def _skip_markup_expression(source: str, index: int) -> int:
    """
    Starting from `index`, skip over a complete markup expression to end of line.

    A markup expression can contain multiple elements:
    - Commands like \\italic, \\huge, \\bold
    - Quoted strings "..."
    - Braced blocks {...}

    This function consumes ALL elements until end of line or a newline.
    This is needed for markup assignments like:
        piuf = _\\markup \\italic "più f"

    Returns the index immediately after the complete markup (or len(source)
    if it hits the end).
    """
    position = index
    length = len(source)

    # Consume all markup elements until we hit a newline or something that's not markup
    while position < length:
        # Skip whitespace (but not newlines - they end the assignment)
        while position < length and source[position] in " \t\r":
            position += 1

        if position >= length or source[position] == "\n":
            break

        char = source[position]

        # Block with braces: {...}
        if char == "{":
            end_index = _grab_balanced(source, position, "{", "}")
            if end_index == -1:
                return length
            position = end_index + 1
            continue

        # Quoted string: "..."
        if char == '"':
            position += 1
            escaped = False
            while position < length:
                current_char = source[position]
                position += 1
                if current_char == '"' and not escaped:
                    break
                escaped = (current_char == "\\") and not escaped
            continue

        # Command: \\something
        if char == "\\":
            position += 1
            while (
                position < length
                and (source[position].isalnum() or source[position] in "_-")
            ):
                position += 1
            continue

        # Scheme: #...
        if char == "#":
            position += 1
            while position < length and not source[position].isspace():
                position += 1
            continue

        # Direction markers: ^ or _
        if char in "^_":
            position += 1
            continue

        # Anything else: consume token and stop
        token_start = position
        while (
            position < length
            and not source[position].isspace()
            and source[position] not in '{}"'
        ):
            position += 1

        # If we consumed a token that's not a markup command, we're done
        if position > token_start:
            token = source[token_start:position]
            # Check if this is a markup-related token
            markup_keywords = ('markup', 'italic', 'bold', 'huge', 'large', 'small', 'center-align', 'center-column')
            is_markup_token = any(kw in token for kw in markup_keywords)
            if not is_markup_token:
                # Not a markup token - backtrack and stop
                position = token_start
                break
        else:
            # No token consumed - skip this character to avoid infinite loop
            position += 1
            break

    return position


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
    text = re.sub(r"\\tu(?!plet)\s*", " ", text)

    text = re.sub(r"\\(?:" + "|".join(unsupported) + r")\b", " ", text)
    
    # Fix scheme errors: ## { # } -> remove these malformed empty scheme blocks
    text = re.sub(r"##\s*\{\s*#\s*\}", "", text)
    
    # DO NOT remove roman numeral variables like \Ibcn, \IIbfn, etc.
    # These are movement-specific music/figuremode assignments that hold actual content.

    # Figured bass broken across lines: "<\n  ->" -> "<->"
    text = re.sub(r"<\s*\n\s*([+\-\d\s]+>)", r"<\1", text)

    # Dotted duration inheritance issues
    # DISABLED: These patterns can accidentally transform valid music sequences
    # text = re.sub(r"(\d+)\.(\([a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s*\))", r"\1\2\3\4\5", text)
    # text = re.sub(r"(\d+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.", r"\1\2\3\4", text)

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

    # Fix malformed figuremode tokens that leave a dangling close brace
    text = re.sub(r"\\figuremode\s*}", "}", text)
    
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

    # Fix missing space in key declarations: \key mi\minor -> \key mi \minor
    text = re.sub(
        r"(\\key\s+(?:do|re|mi|fa|sol|la|si))\\(major|minor)\b",
        r"\1 \\\2",
        text,
    )
    
    # Fix orphaned "r" rest on its own line appearing as note name
    # Pattern: lines that are just "r" or "r8" or "r4" when they should be rests within music
    text = re.sub(r"^\s*r(?:8|4|2|1)?\s*$", "", text, flags=re.MULTILINE)
    
    # Fix \tempo with malformed arguments: "\tempo 2. = 60" (should be "\tempo 2.=60" or removed)
    # Remove tempo marks that have syntax issues
    text = re.sub(r"\\\\tempo\s+[^=]+=\s*\d+", "", text)
    # Remove bare \bar commands that have no argument (invalid LilyPond)
    text = re.sub(r"(?m)^\s*\\bar\s*$", "", text)
    # Final fallback: strip any remaining bare underscores
    text = re.sub(r"_+", " ", text)

    # Remove stray + characters that cling to notes (e.g., la+)
    text = re.sub(r"((?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d?)\+", r"\1", text)
    # Remove invalid "-+" attached to notes (e.g., si4-+)
    text = re.sub(r"((?:do|re|mi|fa|sol|la|si|[a-gr])[',]*\d*\.?)-\+", r"\1", text)
    # Remove leading + tokens at line starts before notes
    text = re.sub(r"(?m)^[ \t]*\+(?=\s*[a-gr])", "", text)
    # Collapse standalone + or - tokens between whitespace
    text = re.sub(r"\s+[\+-](?=\s)", " ", text)
    
    # Fix repeating note patterns: Note^ followed by same/different note (NOTENAME_PITCH error)
    # Pattern: "mi^ la la" -> "mi la la" (remove the ^ articulation that confuses the parser)
    note_token = r"(?:do|re|mi|fa|sol|la|si|[a-gr])[a-z]*"
    text = re.sub(
        rf"(\\b{note_token}[',]*\\d?)\\^\\s+(?={note_token})",
        r"\\1 ",
        text,
    )
    # Also handle no-space caret: "dod^ mi" or "dod^mi" -> "dod mi"
    text = re.sub(
        rf"(\\b{note_token}[',]*\\d?)\\^(?=\\s*{note_token})",
        r"\\1",
        text,
    )
    # Remove caret at end-of-line before next note on following line.
    text = re.sub(rf"(\\b{note_token}[',]*\\d?)\\^\\s*$", r"\\1", text, flags=re.MULTILINE)
    
    # Remove \vspace from middle of markup strings
    text = re.sub(r'"\\\\vspace[^"]*', r'"', text)

    # Split glued Italian note names (e.g., remi2 -> re mi2, faddod, -> fad dod,)
    # Covers common accidentals: dod, red, mid, fad, sold, lad, sid.
    solfege = r"(?:dod|red|mid|fad|sold|lad|sid|do|re|mi|fa|sol|la|si)"
    text = re.sub(
        rf"\\b({solfege})([',]*?)({solfege})([',]*\\d*)\\b",
        r"\\1\\2 \\3\\4",
        text,
    )
    
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
        r"(?m)^\s*\\center-align\b.*$",
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


def _remove_standalone_braced_text_lines(text: str) -> Tuple[str, int]:
    """Remove lines that are just { ... } with no Lily commands (leftover labels)."""
    pattern = re.compile(r"(?m)^\s*\{\s*[^\\{}]*\s*\}\s*$")
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


def _remove_music_inline_strings(text: str) -> Tuple[str, int]:
    """Remove quoted strings that follow musical tokens, even across line breaks."""
    note_re = re.compile(r"\b(?:do|re|mi|fa|sol|la|si|[a-gr]|r)[',]*\d", re.I)
    removed = 0
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    in_str = False

    for line in lines:
        if in_str:
            end = line.find('"')
            if end != -1:
                line = line[end + 1 :]
                in_str = False
                removed += 1
            else:
                removed += 1
                continue

        if '"' in line and note_re.search(line):
            start = line.find('"')
            if start != -1:
                end = line.find('"', start + 1)
                if end != -1:
                    line = line[:start] + line[end + 1 :]
                    removed += 1
                else:
                    line = line[:start]
                    in_str = True
                    removed += 1

        out.append(line)

    cleaned = "".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removed


def _remove_empty_figuremode_blocks(text: str) -> Tuple[str, int]:
    """Remove \\figuremode blocks that only contain layout directives (no figures)."""
    removed = 0

    assign_re = re.compile(r"(?m)^[ \t]*[A-Za-z_][\w-]*\s*=\s*\\figuremode\s*\{")
    inline_re = re.compile(r"\\figuremode\s*\{")

    def has_figures(content: str) -> bool:
        if re.search(r"<[^>]*>", content):
            return True
        return bool(re.search(r"\b\d+\b", content))

    def _strip_at(match_start: int, brace_start: int, is_assignment: bool) -> tuple[int, int] | None:
        brace_end = _grab_balanced(text, brace_start, "{", "}")
        if brace_end == -1:
            return None
        content = text[brace_start + 1:brace_end]
        if has_figures(content):
            return None
        remove_start = match_start if is_assignment else brace_start
        remove_end = brace_end + 1
        while remove_end < len(text) and text[remove_end] == "\n":
            remove_end += 1
        return remove_start, remove_end

    # First pass: assignment-style blocks
    search_start = 0
    while True:
        match = assign_re.search(text, search_start)
        if not match:
            break
        brace_start = match.end() - 1
        removal = _strip_at(match.start(), brace_start, True)
        if removal:
            start, end = removal
            text = text[:start] + text[end:]
            removed += 1
            search_start = start
        else:
            search_start = match.end()

    # Second pass: inline \\figuremode blocks
    search_start = 0
    while True:
        match = inline_re.search(text, search_start)
        if not match:
            break
        brace_start = match.end() - 1
        removal = _strip_at(match.start(), brace_start, False)
        if removal:
            start, end = removal
            text = text[:start] + text[end:]
            removed += 1
            search_start = start
        else:
            search_start = match.end()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def _final_cleanup(text: str) -> str:
    """Dataset-tail cleanup that used to live in the driver.

    Runs after engraving stripping (or directly if engravings are kept).
    """
    # Remove specific paper size directive to avoid mis-tokenizing "a4" as a note.
    text = re.sub(r'(?m)^\s*#\(set-default-paper-size\s+"a4"\)\s*\r?\n?', "", text)
    # Remove empty Scheme calls (e.g., #(set-default-paper-size ))
    text = re.sub(r"#\(\s*[A-Za-z0-9_-]+\s*\)", "", text)

    # Remove stray standalone closing braces
    text = re.sub(r"(?m)^\s*\}\s*$", "", text)

    # Drop malformed assignment lines missing '=' (e.g., "IIvlIni4 do8")
    # IMPORTANT: Only match lines that START with uppercase/underscore (variable names),
    # NOT lowercase (which could be music notes like "sol2 re8 mi4")
    text = re.sub(r"(?m)^[ \t]*[A-Z_][\w-]*[ \t]+[^\s=][^\n]*$", "", text)

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
        "etc",
        "tu",  # Tutti marking (performance/engraving directive)
    )
    text = re.sub(r"\\(?:" + "|".join(unsupported) + r")\b", "", text)

    # Remove Stem transparency overrides (engraving-only commands)
    text = re.sub(r"\\once\s+\\override\s+Stem\s+#'transparent\s+=\s+##t\s*", "", text)
    text = re.sub(r"\\override\s+Stem\s+#'transparent\s+=\s+##t\s*", "", text)
    text = re.sub(r"\\revert\s+Stem\.#'transparent\s*", "", text)
    # Also handle the version already normalized by earlier cleanup (with space after dot)
    text = re.sub(r"\\revert\s+Stem\s+#'transparent\s*", "", text)

    # Remove \parenthesize directive (engraving-only, makes symbols appear in parentheses)
    text = re.sub(r"-?\\parenthesize\s+", "", text)

    # Remove \noBeam directive (engraving-only, prevents automatic beaming)
    # Only remove horizontal whitespace (not newlines) to preserve line structure
    text = re.sub(r"\\noBeam\b[ \t]*", "", text)

    # Remove cautionary/reminder accidentals (? and ! after note names)
    # These are engraving hints to force showing accidentals for clarity
    # Match note name (Italian: do/re/mi/fa/sol/la/si OR English: a-g) + accidentals (d/b/is/es) + octave + duration + ? or !
    text = re.sub(r"(\b(?:do|re|mi|fa|sol|la|si|[a-g])(?:d|b|is|es|isbf|esbf)?[',]*(?:\d+\.*)?)[?!]", r"\1", text, flags=re.I)

    # Strip leftover markup tokens that can survive markup removal and break parsing
    text = re.sub(
        r"\\(?:super|bold|italic|center-align|column|musicglyph|parentSlur|fill-line|smaller|larger)\b",
        "",
        text,
    )
    text = re.sub(r"\\(?:I|II|III|IV)[A-Za-z][\w-]*\b", "", text)
    # Remove stray numeric identifier lines (e.g., leftover movement labels)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s*$", "", text)

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

    # If a top-level assignment appears while still inside an open block,
    # insert missing closing brace(s) before the assignment.
    assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
    lines = text.splitlines()
    out: List[str] = []
    depth = 0
    for line in lines:
        if assign_re.match(line) and depth > 0:
            out.extend(["}"] * depth)
            depth = 0
        out.append(line)
        line_no_comment = line.split("%", 1)[0]
        depth += line_no_comment.count("{") - line_no_comment.count("}")
        if depth < 0:
            depth = 0
    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")

    # Musical dotted patterns / figured bass / revert fixes
    # DISABLED: These patterns can accidentally transform valid music sequences
    # text = re.sub(r"(\d+)\.(\([a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s*\))", r"\1\2\3\4\5", text)
    # text = re.sub(r"(\d+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.(\s+[a-z_]+)\.", r"\1\2\3\4", text)
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


def _remove_footnotes(source: str) -> Tuple[str, int]:
    """
    Remove \\footnote commands and their arguments.

    Footnote syntax: \\footnote #'(offset) \\markup{...} annotated_object

    We need to remove:
    1. The \\footnote keyword
    2. The Scheme offset expression: #'(...)
    3. The markup expression

    This leaves the annotated object (note/rest/etc) intact.

    Returns (cleaned_source, removed_count).
    """
    removed_count = 0
    search_start = 0
    output_parts: List[str] = []

    while True:
        match = RE_FOOTNOTE.search(source, search_start)
        if not match:
            output_parts.append(source[search_start:])
            break

        # Keep everything before the footnote
        output_parts.append(source[search_start:match.start()])
        position = match.end()

        # Skip whitespace
        while position < len(source) and source[position] in " \t":
            position += 1

        # Remove Scheme expression: #'(...)
        if position < len(source) and source[position] == "#":
            position += 1
            # Skip optional quote mark
            if position < len(source) and source[position] == "'":
                position += 1
            # Skip whitespace
            while position < len(source) and source[position] in " \t":
                position += 1
            # Remove balanced parentheses
            if position < len(source) and source[position] == "(":
                end_paren = _grab_balanced(source, position, "(", ")")
                if end_paren != -1:
                    position = end_paren + 1

        # Skip whitespace
        while position < len(source) and source[position] in " \t":
            position += 1

        # Remove the markup expression
        position = _skip_markup_expression(source, position)

        search_start = position
        removed_count += 1

    return "".join(output_parts), removed_count


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
# LilyPond parser-based engraving strip (experimental)
# ---------------------------------------------------------------------------

RE_ASSIGNMENT = re.compile(r"(^|[^\w-])([A-Za-z][\w-]*)\s*=\s*", re.M)


def _find_music_assignments(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Locate named music assignments and return spans for RHS replacement.

    Returns a list of (rhs_start, rhs_end, var_name, rhs_text).
    """
    results: List[Tuple[int, int, str, str]] = []
    search_start = 0
    length = len(text)

    while search_start < length:
        match = RE_ASSIGNMENT.search(text, search_start)
        if not match:
            break

        rhs_start = match.end()
        while rhs_start < length and text[rhs_start].isspace():
            rhs_start += 1

        name = match.group(2)

        # Skip markup assignments (start with _ or ^).
        if rhs_start < length and text[rhs_start] in ("_", "^"):
            search_start = rhs_start + 1
            continue

        # name = \relative ...
        if text.startswith("\\relative", rhs_start):
            rel_match = re.search(r"\\relative\b(?:\s+[^\s{}%]+)?\s*\{", text[rhs_start:], re.I)
            if not rel_match:
                search_start = rhs_start + 1
                continue
            brace_open = rhs_start + rel_match.end() - 1
            brace_close = _grab_balanced(text, brace_open, "{", "}")
            if brace_close == -1:
                search_start = rhs_start + 1
                continue
            rhs_end = brace_close + 1
            results.append((rhs_start, rhs_end, name, text[rhs_start:rhs_end]))
            search_start = rhs_end
            continue

        # name = \transpose ...
        if text.startswith("\\transpose", rhs_start):
            tr_match = re.search(r"\\transpose\b(?:\s+[^\s{}%]+){2}\s*\{", text[rhs_start:], re.I)
            if not tr_match:
                search_start = rhs_start + 1
                continue
            brace_open = rhs_start + tr_match.end() - 1
            brace_close = _grab_balanced(text, brace_open, "{", "}")
            if brace_close == -1:
                search_start = rhs_start + 1
                continue
            rhs_end = brace_close + 1
            results.append((rhs_start, rhs_end, name, text[rhs_start:rhs_end]))
            search_start = rhs_end
            continue

        # name = { ... }
        if rhs_start < length and text[rhs_start] == "{":
            brace_close = _grab_balanced(text, rhs_start, "{", "}")
            if brace_close == -1:
                search_start = rhs_start + 1
                continue
            rhs_end = brace_close + 1
            results.append((rhs_start, rhs_end, name, text[rhs_start:rhs_end]))
            search_start = rhs_end
            continue

        # name = << ... >>
        if text.startswith("<<", rhs_start):
            angle_close = _grab_angles(text, rhs_start)
            if angle_close == -1:
                search_start = rhs_start + 1
                continue
            results.append((rhs_start, angle_close, name, text[rhs_start:angle_close]))
            search_start = angle_close
            continue

        search_start = rhs_start + 1

    return results


def _variable_contains_music(rhs_content: str) -> bool:
    """
    Check if a variable assignment contains actual musical content (notes/rests).

    Returns True if the variable has notes that should be kept for training.
    Returns False if it's pure engraving/layout that can be safely removed.

    Examples:
        "{ c4 d e f }" → True (has notes)
        "{ \\override Stem.length = #7 }" → False (only engraving)
        "^\\markup {Solo}" → False (only markup)
        "{ \\time 4/4 \\key c \\major s1 }" → False (only layout, no actual notes)
        "forma = { \\time 4/4 s1*10 }" → False (layout/spacers, removed after splitting)
        "IvlI = { \\global << \\forma>> }" → False (only references, no notes)
    """
    # Check for actual note/rest tokens (but NOT spacer notes 's')
    # IMPORTANT: Require a digit after the note to avoid matching variable names like "forma", "melodia"
    # Matches: do4, re8, mi16, c'4, g,,2, r4, fad8, etc.
    note_pattern = r'\b(?:do|re|mi|fa|sol|la|si|[a-g]|r)[is|es|isbf|esbf]*[,\']*\d'
    if re.search(note_pattern, rhs_content, re.I):
        return True

    # Check for chord notation with actual pitches: <note ... note>
    # Must have at least one note with duration inside the brackets
    if re.search(r'<[^>]*\b(?:do|re|mi|fa|sol|la|si|[a-g])[is|es|isbf|esbf]*[,\']*\d[^>]*>', rhs_content, re.I):
        return True

    # If we get here, it's likely just engraving/layout/references
    return False


def _find_all_variable_assignments(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Find ALL variable assignments including markup, not just music variables.

    Similar to _find_music_assignments but doesn't skip markup assignments.
    Returns: list of (assign_start, assign_end, var_name, full_assignment_text)
    """
    results: List[Tuple[int, int, str, str]] = []
    search_start = 0
    length = len(text)

    while search_start < length:
        match = RE_ASSIGNMENT.search(text, search_start)
        if not match:
            break

        name = match.group(2)
        assign_start = match.start() if match.group(1) else match.start(2)

        rhs_start = match.end()
        while rhs_start < length and text[rhs_start].isspace():
            rhs_start += 1

        if rhs_start >= length:
            search_start = rhs_start
            continue

        # Handle different assignment types
        assign_end = None

        # Simple markup: name = ^\markup ...
        if text[rhs_start] in ("_", "^"):
            # Find end of markup
            markup_start = rhs_start
            markup_start += 1  # skip _ or ^
            while markup_start < length and text[markup_start].isspace():
                markup_start += 1

            if text.startswith("\\markup", markup_start):
                # Skip \markup and find the braces or quoted string
                markup_start += 7  # len("\\markup")
                while markup_start < length and text[markup_start].isspace():
                    markup_start += 1

                if markup_start < length and text[markup_start] == '{':
                    brace_end = _grab_balanced(text, markup_start, "{", "}")
                    assign_end = brace_end + 1 if brace_end != -1 else markup_start + 1
                else:
                    # Find end of line
                    assign_end = text.find('\n', markup_start)
                    if assign_end == -1:
                        assign_end = length
            else:
                # Just direction marker, find end of line
                assign_end = text.find('\n', rhs_start)
                if assign_end == -1:
                    assign_end = length

        # Block assignment: name = { ... }
        elif text[rhs_start] == '{':
            brace_end = _grab_balanced(text, rhs_start, "{", "}")
            assign_end = brace_end + 1 if brace_end != -1 else rhs_start + 1

        # \relative, \transpose, etc.
        elif text.startswith("\\relative", rhs_start) or text.startswith("\\transpose", rhs_start):
            # Find the opening brace
            brace_pos = text.find('{', rhs_start)
            if brace_pos != -1:
                brace_end = _grab_balanced(text, brace_pos, "{", "}")
                assign_end = brace_end + 1 if brace_end != -1 else brace_pos + 1
            else:
                assign_end = text.find('\n', rhs_start)
                if assign_end == -1:
                    assign_end = length

        # Angle brackets: name = << ... >>
        elif text.startswith("<<", rhs_start):
            angle_end = _grab_angles(text, rhs_start)
            assign_end = angle_end if angle_end != -1 else rhs_start + 2

        if assign_end is not None:
            results.append((
                assign_start,
                assign_end,
                name,
                text[assign_start:assign_end]
            ))
            search_start = assign_end
        else:
            search_start = rhs_start + 1

    return results


def _remove_engraving_only_variables(text: str) -> Tuple[str, int]:
    """
    Remove variable assignments that contain ONLY engraving/layout, no actual music.

    This is a coarse-grained filter that operates at the variable level:
    - If a variable has NO notes/rests → remove entire variable
    - If a variable has notes → keep it (even if it also has engraving)

    This prevents the "broken Scheme code" problem by removing entire
    engraving-only blocks rather than trying to surgically edit them.

    Returns (cleaned_text, removed_count).
    """
    assignments = _find_all_variable_assignments(text)
    if not assignments:
        return text, 0

    removed_count = 0

    # Work backwards so indices remain valid
    for assign_start, assign_end, name, full_text in reversed(assignments):
        if not _variable_contains_music(full_text):
            # Find the start of the line
            line_start = assign_start
            while line_start > 0 and text[line_start - 1] not in '\n':
                line_start -= 1

            # Remove the entire assignment (including trailing newline if present)
            end_pos = assign_end
            if end_pos < len(text) and text[end_pos] == '\n':
                end_pos += 1

            text = text[:line_start] + text[end_pos:]
            removed_count += 1

    return text, removed_count


def _remove_engraving_only_paragraphs(text: str) -> Tuple[str, int]:
    """
    Remove paragraphs (text blocks separated by blank lines) that contain
    ONLY engraving/layout commands, no actual musical content.

    IMPORTANT: Variable assignments are handled by _remove_engraving_only_variables()
    and should be skipped here to avoid splitting them incorrectly.

    A paragraph is defined as text between two consecutive newlines.

    Examples of paragraphs that will be removed:
        - \\pointAndClickOff
        - \\paper { ... }
        - #(set-global-staff-size 17)
        - \\override Score.MetronomeMark.transparent = ##t

    Examples of paragraphs that will be kept:
        - melody = { c4 d e f }
        - { c4 d e f }  (even without variable name)
        - \\time 4/4 s1*10  (has spacer notes)

    Returns (cleaned_text, removed_count).
    """
    # First, find all variable assignments to avoid splitting them
    assignments = _find_all_variable_assignments(text)
    assignment_ranges = [(start, end) for start, end, _, _ in assignments]

    def is_inside_assignment(pos):
        """Check if position is inside a variable assignment."""
        for start, end in assignment_ranges:
            if start <= pos < end:
                return True
        return False

    # Split into paragraphs (separated by blank lines - two consecutive newlines)
    # Keep the separators so we can rebuild
    paragraphs = re.split(r'(\n\s*\n)', text)

    result_parts = []
    removed_count = 0
    current_pos = 0

    for i, para in enumerate(paragraphs):
        para_start = current_pos
        para_end = current_pos + len(para)
        current_pos = para_end

        # If this is a separator (blank line), keep it conditionally
        if re.match(r'^\n\s*\n$', para):
            # Only keep if we have content before and after
            if result_parts and i < len(paragraphs) - 1:
                result_parts.append('\n\n')  # Normalize to double newline
            continue

        # Skip empty paragraphs
        if not para.strip():
            continue

        # Skip paragraphs that are part of a variable assignment
        # (they're already handled by _remove_engraving_only_variables)
        if is_inside_assignment(para_start):
            result_parts.append(para)
            continue

        # Check if paragraph contains music
        if _variable_contains_music(para):
            result_parts.append(para)
        else:
            # Keep \language because pitch names depend on it (e.g., do/re/mi).
            if re.search(r"\\language\b", para, re.I):
                result_parts.append(para)
                continue
            # Check if it's a comment-only paragraph (should keep)
            lines = para.strip().split('\n')
            all_comments = all(line.strip().startswith('%') or not line.strip() for line in lines)

            if all_comments and any(line.strip() for line in lines):
                # Keep comment paragraphs
                result_parts.append(para)
            else:
                # Remove this paragraph - it's pure engraving/layout
                removed_count += 1

    return '\n\n'.join(result_parts), removed_count


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

    If opts.keep_engraving is True, retain engraving and apply final cleanup.
    If False, strip engraving-only variables/paragraphs, then apply final cleanup.
    """
    messages: list[str] = []

    def _add_count(label: str, count: int) -> None:
        if count:
            messages.append(f"{label}={count}")

    if getattr(opts, "keep_engraving", True):
        messages.append("mode=keep")
        cleaned = text
        cleaned = _final_cleanup(cleaned)
    else:
        messages.append("mode=strip")
        cleaned = text
        cleaned, engrave_var_count = _remove_engraving_only_variables(cleaned)
        _add_count("engrave_vars", engrave_var_count)

        cleaned, engrave_para_count = _remove_engraving_only_paragraphs(cleaned)
        _add_count("engrave_paras", engrave_para_count)

        cleaned = _final_cleanup(cleaned)

    if messages:
        print("[engrave_strip] " + " ".join(messages), file=sys.stderr)

    return cleaned
