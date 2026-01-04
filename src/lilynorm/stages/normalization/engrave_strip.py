from __future__ import annotations

import re
import sys
from typing import Tuple, List, Dict

def _grab_balanced(
    text: str,
    start: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int:
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

def _remove_metadata_headers(text: str) -> Tuple[str, int]:
    # Drop header-style metadata lines, keep \language for pitch names.
    removed_count = 0

    # Remove \version lines
    pattern_version = re.compile(r"(?m)^\s*\\version\b.*$")
    cleaned, count = pattern_version.subn("", text)
    removed_count += count

    # Remove metadata header lines like: { "GPT231020 baroquemusic.it""Title""License"}
    # These appear after \version and before \language
    # Pattern: line starting with { followed by quoted strings
    pattern_metadata = re.compile(r'(?m)^\s*\{\s*"[^"]*"[^}]*\}\s*$')
    cleaned, count = pattern_metadata.subn("", cleaned)
    removed_count += count

    return cleaned, removed_count

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

RE_SPACER_ONLY_SUBVOICE = re.compile(
    r"(?sx)"
    r"(\\\\\{)"                        # subvoice start
    r"\s*(?:s[0-9.']*(?:\s+|$))+"       # one or more spacer durations
    r"\s*(\})"                          # closing brace
)

def _skip_markup_expression(source: str, index: int) -> int:
    # Walk a markup expression starting at index and return the first position after it.
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

def _remove_non_whitelisted_commands(text: str) -> Tuple[str, int]:
    WHITELIST = {
        'relative',
        'absolute',
        'time',
        'key',
        'partial',
        'repeat',
        'alternative',
        'tuplet',
    }

    pattern = r'\\([a-zA-Z]+)\b'
    removal_count = 0

    def replace_if_not_whitelisted(match):
        nonlocal removal_count
        command_name = match.group(1)
        if command_name in WHITELIST:
            return match.group(0)
        else:
            removal_count += 1
            return ''

    result = re.sub(pattern, replace_if_not_whitelisted, text)
    return result, removal_count

def _final_cleanup(text: str) -> Tuple[str, int]:
    text = re.sub(r'(?m)^\s*#\(set-default-paper-size\s+"a4"\)\s*\r?\n?', "", text)
    # Remove empty Scheme calls (e.g., #(set-default-paper-size ))
    text = re.sub(r"#\(\s*[A-Za-z0-9_-]+\s*\)", "", text)

    # Remove Scheme string literals: #"..." (e.g., #"scripts.ufermata", MIDI instrument names)
    text = re.sub(r'#"[^"]*"', "", text)

    # Remove Scheme number literals (e.g., #0.5, #-1.2, #3)
    text = re.sub(r'#[+-]?\d+(?:\.\d+)?', '', text)

    # Remove Scheme empty block expressions: #{ } or #{\s*}
    text = re.sub(r'#\{\s*\}', "", text)

    # Remove empty braces
    text = re.sub(r' ?\{[ \t]*\}', "", text)

    # Remove leftover alignment directives
    text = re.sub(r'[\^_-]*-?align\b', "", text)

    # Drop malformed assignment lines missing '='
    text = re.sub(r"(?m)^[ \t]*[A-Z_][\w-]*[ \t]+[^\s=][^\n]*$", "", text)

    unsupported = (
        "mbreak",
        "trasp",
        "notrasp",
        "typeset",
        "notypeset",
        "Voice",
        "terzine",
        "con",
        "senza",
        "notrasp",
        "etc",
        "tu",
    )
    text = re.sub(r"\\(?:" + "|".join(unsupported) + r")\b", "", text)

    text = re.sub(r"\\once\s+\\override\s+Stem\s+#'transparent\s+=\s+##t\s*", "", text)
    text = re.sub(r"\\override\s+Stem\s+#'transparent\s+=\s+##t\s*", "", text)
    text = re.sub(r"\\revert\s+Stem\.#'transparent\s*", "", text)
    text = re.sub(r"\\revert\s+Stem\s+#'transparent\s*", "", text)

    text = re.sub(r"-?\\parenthesize\s+", "", text)
    text = re.sub(r"\\noBeam\b[ \t]*", "", text)
    text = re.sub(r"(\b(?:do|re|mi|fa|sol|la|si|[a-g])(?:d|b|is|es|isbf|esbf)?[',]*(?:\d+\.*)?)[?!]", r"\1", text, flags=re.I)

    text = re.sub(r'[\^_-]\\markup\s*"[^"]*"', '', text)
    text = re.sub(r'[\^_-]\\markup\s*\{[^{}]*\}', '', text)
    text = re.sub(r'[\^_-]\\markup\s*\\[A-Za-z]+\s*"[^"]*"', '', text)

    text = re.sub(r'(?:-?\s*[XY]-offset\s*#\s*[+-]?\d+(?:\.\d+)?)+', '', text)
    text = re.sub(r'[\^_]"[^"]*"', '', text)
    text = re.sub(r'[\^_]\s*\{[^}]*\}', '', text)

    # Remove standalone dynamic markings
    text = re.sub(r'\b(pp?p?p?|ff?f?f?|mf|mp|sf|sfz|rfz|fz)\b', '', text)

    # Remove standalone quoted strings (NOT preceded by #)
    text = re.sub(r'(?<!#)"[^"]*"', '', text)

    text = re.sub(r'\\grace\s+(?:\S+|\{[^}]*\})\s*', '', text)
    text = re.sub(r'\\appoggiatura(?:\s+\S+|\{[^}]*\})\s*', '', text)

    text = re.sub(r"([a-z]+)\s+([',]+)", r'\1\2', text)
    text = re.sub(r'-[!.+^_-]', '', text)
    text = re.sub(r'\\~', '', text)
    text = re.sub(r'\\\(([^)\\]*)\)', r'(\1)', text)
    text = re.sub(r'\(([^)\\]*)\\\)', r'(\1)', text)
    text = re.sub(r'\\\[([^\]\\]*)\]', r'[\1]', text)
    text = re.sub(r'\[([^\]\\]*)\\\]', r'[\1]', text)
    # Remove \tr trill marks - don't use \b because \tr can appear without spaces: fa'\trfa8
    text = re.sub(r'\\tr', '', text)
    text = re.sub(r'\?', '', text)
    text = re.sub(r'\\[<>!,]', '', text)
    text = re.sub(r'\bmbreak\b', '', text)

    # Remove Staff context blocks: "Staff  {  VerticalAxisGroup... } { \key ..."
    # Match Staff followed by its property block, then replace with just the opening brace
    # This must come BEFORE the Staff.property removal to avoid partial matches
    # Use .*? non-greedy match to handle nested braces in property values
    before_staff = text.count('Staff  {')
    text = re.sub(r'Staff\s+\{.*?\}\s*(\{)', r'\1', text)
    after_staff = text.count('Staff  {')
    if before_staff > 0:
        import sys
        print(f"[DEBUG] Staff blocks before:{before_staff} after:{after_staff}", file=sys.stderr)

    # Remove Staff.midiInstrument = lines and other Staff settings
    text = re.sub(r'\s*Staff\.midiInstrument\s*=\s*', ' ', text)
    text = re.sub(r'\s*Staff\.\w+\s*', ' ', text)

    text = re.sub(r'\\(?:stemUp|stemDown|stemNeutral|slurUp|slurDown|slurNeutral|tieUp|tieDown|tieNeutral|shiftOn|shiftOff|shiftOnn|shiftOnnn)\b', '', text)

    text = re.sub(r'\\once\s*', '', text)
    with_blocks = []
    placeholder_template = "<<<WITH_BLOCK_{}>>>"

    result = []
    i = 0
    while i < len(text):
        # Check for \with
        if text[i:i+5] == r'\with':
            # Find opening brace
            j = i + 5
            while j < len(text) and text[j] in ' \t\n':
                j += 1

            if j < len(text) and text[j] == '{':
                # Balance braces to find the end of the \with block
                brace_count = 1
                k = j + 1
                while k < len(text) and brace_count > 0:
                    if text[k] == '{':
                        brace_count += 1
                    elif text[k] == '}':
                        brace_count -= 1
                    k += 1

                # Save this \with block
                with_block = text[i:k]
                idx = len(with_blocks)
                with_blocks.append(with_block)
                result.append(placeholder_template.format(idx))
                i = k
                continue

        result.append(text[i])
        i += 1

    text = ''.join(result)

    # Remove inline \override commands
    # Updated pattern handles all value types:
    # - Simple numbers: 1, -3
    # - Hash values: #3, #-0, ##t, ##f
    # - Complex values: #'(), #'(-20 . +2), #'property
    text = re.sub(
        r"\\override\s+\S+(?:\.[^\s=]+)*(?:\s+#'[\w\-]+)?\s*=\s*(?:#'?\([\s\w\d.+\-]*\)|#[#']?[\w\-+.]+|\d+)\s*",
        '',
        text
    )

    # Restore \with blocks
    for idx, with_block in enumerate(with_blocks):
        text = text.replace(placeholder_template.format(idx), with_block)

    # Remove \revert commands (reverse a previous \override)
    # Pattern: \revert Context.Property
    text = re.sub(r'\\revert\s+\S+', '', text)

    # Remove \set Staff commands (selective - only engraving-related ones)
    # Keep: \set Staff.midiInstrument (affects MIDI playback - musical)
    # Remove: \set Staff.ottavation, \set Staff.instrumentName (visual labels)
    # Pattern: \set Staff.X where X is not midiInstrument
    text = re.sub(r'\\set\s+Staff\.(?!midiInstrument\b)\S+\s*=\s*[^\n\\]+', '', text)

    # =========================================================================

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

    # Remove incomplete/malformed \language lines (missing value)
    text = re.sub(r'(?m)^\\language\s*$', '', text)

    # Apply whitelist filter to remove all non-essential backslash commands
    # This catches any remaining dynamics, articulations, performance marks, etc.
    text, whitelist_removed = _remove_non_whitelisted_commands(text)

    # Remove trailing _ or ^ after notes (must come AFTER whitelist filter)
    text = re.sub(r"([a-z]+[',]*\d*\.*)\s*[\^_]+(?=\s|[)\]]|$)", r'\1', text)

    # Remove Y-offset and X-offset text
    text = re.sub(r'\s*[XY]-offset\s*', ' ', text)

    # Fix extra spaces before dots in note durations (e.g., "sib  ." -> "sib.")
    text = re.sub(r'([a-z]+[,\']*)\s+\.', r'\1.', text)

    text = re.sub(r' ?\{\s*\}', "", text)
    text = re.sub(r'(?:\s*-\s*)+(?=\s|$)', ' ', text)

    # Remove orphaned backslashes (from removed commands)
    text = re.sub(r'\\\s+', ' ', text)

    return text, whitelist_removed

RE_ASSIGNMENT = re.compile(r"(^|[^\w-])([A-Za-z][\w-]*)\s*=\s*", re.M)

def _variable_contains_music(rhs_content: str) -> bool:
    # Heuristic: return True if the assignment includes notes/rests or chords.
    # Remove \incipit blocks first (they contain visual-only notes that aren't real music)
    # Pattern: \incipit { ... } where braces may be nested
    content = rhs_content
    while True:
        # Find \incipit followed by a brace block
        match = re.search(r'\\incipit\s*\{', content)
        if not match:
            break

        brace_start = match.end() - 1  # Position of opening brace
        brace_end = _grab_balanced(content, brace_start, "{", "}")

        if brace_end != -1:
            # Remove the entire \incipit { ... } block
            content = content[:match.start()] + content[brace_end + 1:]
        else:
            # Couldn't find matching brace, just remove the \incipit keyword
            content = content[:match.start()] + content[match.end():]
            break

    # Check for actual note/rest tokens (but NOT spacer notes 's')
    # Require a digit after the note to avoid matching variable names like "forma", "melodia"
    # Matches: do4, re8, mi16, c'4, g,,2, r4, fad8, etc.
    note_pattern = r'\b(?:do|re|mi|fa|sol|la|si|[a-g]|r)[is|es|isbf|esbf]*[,\']*\d'
    if re.search(note_pattern, content, re.I):
        return True

    # Check for chord notation with actual pitches: <note ... note>
    # Duration can be inside the chord (e.g., <do4 mi4 sol4>) OR after closing > (e.g., <do mi sol>4)
    # Pattern: < ... at least one note name ... > with optional duration after
    # Use negative lookbehind/lookahead to avoid matching << >> (simultaneous music delimiters)
    chord_pattern = r'(?<![<>])<[^<>]*\b(?:do|re|mi|fa|sol|la|si|[a-g])[is|es|isbf|esbf]*[,\']*\d?[^<>]*>(?![<>])(?:\d+)?'
    if re.search(chord_pattern, content, re.I):
        return True

    # If we get here, it's likely just engraving/layout/references
    return False

def _find_all_variable_assignments(text: str) -> List[Tuple[int, int, str, str]]:
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
    # Drop assignments that contain only engraving/layout and no notes.
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
    # First, find all variable assignments to avoid splitting them
    assignments = _find_all_variable_assignments(text)
    assignment_ranges = [(start, end) for start, end, _, _ in assignments]

    def is_inside_assignment(pos):
        # Check if position is inside a variable assignment.
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

def _remove_engraving_only_top_level(text: str) -> Tuple[str, int]:
    # Find all variable assignments to avoid processing them here
    assignments = _find_all_variable_assignments(text)
    assignment_ranges = [(start, end) for start, end, _, _ in assignments]

    def is_inside_assignment(pos):
        # Check if position is inside a variable assignment.
        for start, end in assignment_ranges:
            if start <= pos < end:
                return True
        return False

    # Split into lines for line-by-line processing
    lines = text.split('\n')
    result_lines = []
    removed_count = 0
    current_pos = 0

    for line in lines:
        line_start = current_pos
        line_end = current_pos + len(line) + 1  # +1 for newline
        current_pos = line_end

        # Skip empty lines (keep them for structure)
        if not line.strip():
            result_lines.append(line)
            continue

        # Skip lines that are part of a variable assignment
        # (they're already handled by _remove_engraving_only_variables)
        if is_inside_assignment(line_start):
            result_lines.append(line)
            continue

        # Keep essential directives
        if re.search(r'\\version\b|\\language\b', line, re.I):
            result_lines.append(line)
            continue

        # Keep comment lines
        if line.strip().startswith('%'):
            result_lines.append(line)
            continue

        # Check if line contains music
        if _variable_contains_music(line):
            result_lines.append(line)
        else:
            # This is an engraving-only line - remove it
            removed_count += 1

    return '\n'.join(result_lines), removed_count

try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        # Minimal fallback for the keep_engraving flag.
        keep_engraving: bool = True  # default: keep engravings

def run(text: str, opts: NormOptions) -> str:
    # Pipeline entry point: keep or strip engraving, then do a final cleanup.
    messages: list[str] = []

    def _add_count(label: str, count: int) -> None:
        if count:
            messages.append(f"{label}={count}")

    if getattr(opts, "keep_engraving", True):
        messages.append("mode=keep")
        cleaned = text
        cleaned, whitelist_count = _final_cleanup(cleaned)
        _add_count("whitelist", whitelist_count)
    else:
        messages.append("mode=strip")
        cleaned = text

        # Remove metadata headers first (version, GPT/AS headers)
        cleaned, metadata_count = _remove_metadata_headers(cleaned)
        _add_count("metadata", metadata_count)

        cleaned, engrave_var_count = _remove_engraving_only_variables(cleaned)
        _add_count("engrave_vars", engrave_var_count)

        cleaned, engrave_para_count = _remove_engraving_only_paragraphs(cleaned)
        _add_count("engrave_paras", engrave_para_count)

        cleaned, engrave_lines_count = _remove_engraving_only_top_level(cleaned)
        _add_count("engrave_lines", engrave_lines_count)

        cleaned, whitelist_count = _final_cleanup(cleaned)
        _add_count("whitelist", whitelist_count)

    if messages:
        print("[engrave_strip] " + " ".join(messages), file=sys.stderr)

    return cleaned
