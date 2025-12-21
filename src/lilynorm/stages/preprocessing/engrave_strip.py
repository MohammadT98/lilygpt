from __future__ import annotations

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
    Replace `name = \\lyricmode { ... }` with `name = {}` (empty block),
    preserving the assignment prefix and structure.

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

        prefix = match.group(1)
        text = text[:match.start()] + prefix + "{}" + text[brace_close_index + 1 :]
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

DYNAMICS = (
    "ppppp|pppp|ppp|pp|p|mp|mf|f|ff|fff|ffff|fffff|fp|sf|sfz|sffz|rfz|fz|sfp|sff|sfpp|sfzp"
)
RE_DYNAMICS = re.compile(rf"(?:[-_^]\s*)?\\(?:{DYNAMICS})\b", re.I)

RE_HAIRPINS = re.compile(
    r"\\[<>!]|\\(?:cresc|decresc|decr|dim|crescendo|diminuendo)\b", re.I
)
RE_ATTACHED_QUOTES = re.compile(r"(?:[-_^]\s*)\"[^\"]*\"")
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
        if inner_content.strip() == "":
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

    This handles nested braces, quoted strings, commands, and Scheme chunks.
    Returns the index immediately after the markup expression (or len(source)
    if it hits the end).
    """
    position = index
    length = len(source)

    while position < length:
        char = source[position]

        # Skip whitespace
        if char in " \t\r\n":
            position += 1
            continue

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

        # Fallback: plain word or token
        token_start = position
        while (
            position < length
            and not source[position].isspace()
            and source[position] not in '{}"'
        ):
            position += 1

        if position == token_start:
            position += 1

    return position


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
    }

    # Attached quotes like ^"text"
    if options.remove_quotes:
        text, removed_quotes = RE_ATTACHED_QUOTES.subn("", text)
        counts["quotes"] += removed_quotes

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
        # Remove \\with { ... } blocks
        updated_text, removed_with_blocks = _remove_with_blocks(text)
        if removed_with_blocks:
            text = updated_text
            counts["overrides"] += removed_with_blocks

        # Remove layout/paper/header blocks
        for directive in ("layout", "paper", "header"):
            updated_text, removed_blocks = _remove_block_directive(text, directive)
            if removed_blocks:
                text = updated_text
                counts["overrides"] += removed_blocks

        # Inline overrides/tweaks/shape/omit...
        for regex in RE_OVERRIDES:
            updated_text, removed = regex.subn("", text)
            if removed:
                text = updated_text
                counts["overrides"] += removed

    # Dynamics & hairpins
    if options.remove_dynamics:
        text, removed_dynamics = RE_DYNAMICS.subn("", text)
        counts["dynamics"] += removed_dynamics

    if options.remove_hairpins:
        text, removed_hairpins = RE_HAIRPINS.subn("", text)
        counts["hairpins"] += removed_hairpins

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
    text = RE_EMPTY_BLOCK_LINE.sub("", text)
    text = RE_EMPTY_ASSIGNMENT_LINE.sub("", text)
    text = RE_INLINE_EMPTY_BRACES.sub(" ", text)
    text = RE_INCLUDE_TAG.sub("", text)
    text = RE_REPEATED_INCLUDE.sub("", text)
    text = RE_EMPTY_SCORES.sub("", text)
    text = RE_EMPTY_LAYOUT_BLOCK.sub("", text)

    if DROP_EMPTY_ASSIGNMENTS:
        text = re.sub(r"(?m)^\s*\w+\s*=\s*\{\s*\}\s*$", "", text)

    if PRUNE_SPACER_SUBVOICES:
        text = _prune_spacer_only_subvoices(text)

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
    """
    Main entry point used by the lilynorm pipeline.

    If opts.keep_engraving is True, the function returns the input unchanged.
    Otherwise it strips engraving information (overrides, markups, etc.) and
    compacts the LilyPond source.
    """
    if getattr(opts, "keep_engraving", True):
        print("[engrave_strip] keeping engravings", file=sys.stderr)
        return text

    # When we are actively stripping engravings, also drop empty assignments.
    global DROP_EMPTY_ASSIGNMENTS
    DROP_EMPTY_ASSIGNMENTS = True

    strip_options = StripOptions(
        remove_overrides=True,
        remove_markups=True,
        remove_marks=True,
        remove_dynamics=True,
        remove_hairpins=True,
        remove_quotes=True,
        space_mode=DEFAULT_SPACE_MODE,
    )

    cleaned, counts = clean_lilypond(text, strip_options)
    print(
        f"[engrave_strip] overrides:{counts['overrides']} "
        f"markups:{counts['markups']} marks:{counts['marks']} "
        f"dynamics:{counts['dynamics']} hairpins:{counts['hairpins']} "
        f"quotes:{counts['quotes']}",
        file=sys.stderr,
    )
    return cleaned