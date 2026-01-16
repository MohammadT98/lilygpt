from __future__ import annotations

import re
import sys

from .utils import grab_balanced, grab_angles


def _remove_metadata_headers(text: str) -> tuple[str, int]:
    text, count1 = re.subn(r"(?m)^\s*\\version\b.*$", "", text)
    text, count2 = re.subn(r'(?m)^\s*\{\s*"[^"]*"[^}]*\}\s*$', "", text)
    return text, count1 + count2


def _split_inline_assignments(text: str) -> str:
    text = re.sub(r"}\s+(?=[A-Za-z_][\w-]*\s*=)", "}\n\n", text)
    text = re.sub(r"\)\s+(?=[A-Za-z_][\w-]*\s*=)", ")\n\n", text)
    return text


def _remove_block_commands(text: str) -> str:
    """Remove commands that take brace-delimited blocks (e.g., \\incipit{...})"""
    # Commands that should be removed along with their {...} blocks
    block_commands = [
        'incipit',
        'with',
        'figures',
        'figuremode',
    ]

    for cmd in block_commands:
        pattern = r'\\' + cmd + r'\s*\{'
        while True:
            match = re.search(pattern, text)
            if not match:
                break

            # Find matching closing brace
            brace_start = match.end() - 1
            brace_end = grab_balanced(text, brace_start, "{", "}")

            if brace_end != -1:
                # Remove entire \\command{...} block
                text = text[:match.start()] + text[brace_end + 1:]
            else:
                # Malformed - just remove the command
                text = text[:match.start()] + text[match.end():]
                break

    # Commands that take a single word argument (e.g., \clef soprano)
    # Remove both the command and its argument
    text = re.sub(r'\\clef\s+\w+', '', text)

    return text


def _remove_non_whitelisted_commands(text: str) -> tuple[str, int]:
    # Musical modes for key signatures
    modes = {'major', 'minor', 'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian'}
    # Core musical commands to preserve
    # NOTE: key/time/tempo are kept in whitelist - corrupted structural lines are removed later in postprocessing
    whitelist = {'relative', 'absolute', 'time', 'key', 'tempo', 'partial', 'repeat', 'alternative', 'tuplet', 'bar'} | modes
    count = 0

    def replace(m):
        nonlocal count
        if m.group(1) in whitelist:
            return m.group(0)
        count += 1
        return ''

    # First remove block commands (commands with braces)
    text = _remove_block_commands(text)

    # Then remove other non-whitelisted commands
    return re.sub(r'\\([a-zA-Z]+)\b', replace, text), count


def _final_cleanup(text: str) -> tuple[str, int]:
    text = re.sub(r'(?m)^\s*#\(set-default-paper-size\s+"a4"\)\s*\r?\n?', "", text)
    text = re.sub(r"#\(\s*[A-Za-z0-9_-]+\s*\)", "", text)
    text = re.sub(r'#"[^"]*"', "", text)
    text = re.sub(r'#[+-]?\d+(?:\.\d+)?', '', text)
    text = re.sub(r'#\{\s*\}', "", text)
    text = re.sub(r' ?\{[ \t]*\}', "", text)
    text = re.sub(r'[\^_-]*-?align\b', "", text)
    text = re.sub(r"(?m)^[ \t]*[A-Z_][\w-]*[ \t]+[^\s=][^\n]*$", "", text)

    unsupported = (
        "mbreak",
        "trasp",
        "notrasp",
        "typeset",
        "notypeset",
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

    text = re.sub(r'\\grace\s+(?:\S+|\{[^}]*\})\s*', ' ', text)
    text = re.sub(r'\\appoggiatura(?:\s+\S+|\{[^}]*\})\s*', ' ', text)
    text = re.sub(
        r'[-_^]?\s*\\(?:staccatissimo|staccato|tenuto|accent|marcato|portato|fermata)',
        ' ',
        text,
    )

    text = re.sub(r"([a-z]+)\s+([',]+)", r'\1\2', text)
    text = re.sub(r'-[!.+^_-]', '', text)
    text = re.sub(r'\\~', '', text)
    text = re.sub(r'\\\(([^)\\]*)\)', r'(\1)', text)
    text = re.sub(r'\(([^)\\]*)\\\)', r'(\1)', text)
    text = re.sub(r'\\\[([^\]\\]*)\]', r'[\1]', text)
    text = re.sub(r'\[([^\]\\]*)\\\]', r'[\1]', text)
    text = re.sub(r'\\tr', '', text)
    text = re.sub(r'\?', '', text)
    text = re.sub(r'\\[<>!,]', '', text)
    text = re.sub(r'\bmbreak\b', '', text)

    text = re.sub(r'Staff\s+\{.*?\}\s*(\{)', r'\1', text)

    text = re.sub(r'\s*Staff\.midiInstrument\s*=\s*', ' ', text)
    text = re.sub(r'\s*Staff\.\w+\s*', ' ', text)
    text = re.sub(r'(?m)^\s*Staff\s*$', '', text)
    text = re.sub(r'(?m)^\s*alignAboveContext\s*=.*$', '', text)
    text = re.sub(r'(?m)^.*StaffSymbol\s*=.*$', '', text)
    text = re.sub(r'StaffSymbol\s*=\s*#\([^)]*\)', '', text)
    text = re.sub(r'(?m)^\s*[A-Za-z_][\w-]*\.[\w.-]+\s*=\s*$', '', text)
    text = re.sub(r'(?m)^\s*TupletBracket\s*=\s*$', '', text)

    text = re.sub(r'\\(?:stemUp|stemDown|stemNeutral|slurUp|slurDown|slurNeutral|tieUp|tieDown|tieNeutral|shiftOn|shiftOff|shiftOnn|shiftOnnn)\b', '', text)

    text = re.sub(r'\\once\s*', '', text)
    with_blocks = []
    placeholder_template = "<<<WITH_BLOCK_{}>>>"

    result = []
    i = 0
    while i < len(text):
        if text[i:i+5] == r'\with':
            j = i + 5
            while j < len(text) and text[j] in ' \t\n':
                j += 1

            if j < len(text) and text[j] == '{':
                brace_count = 1
                k = j + 1
                while k < len(text) and brace_count > 0:
                    if text[k] == '{':
                        brace_count += 1
                    elif text[k] == '}':
                        brace_count -= 1
                    k += 1

                with_block = text[i:k]
                idx = len(with_blocks)
                with_blocks.append(with_block)
                result.append(placeholder_template.format(idx))
                i = k
                continue

        result.append(text[i])
        i += 1

    text = ''.join(result)

    text = re.sub(
        r"\\override\s+\S+(?:\.[^\s=]+)*(?:\s+#'[\w\-]+)?\s*=\s*(?:#'?\([\s\w\d.+\-]*\)|#[#']?[\w\-+.]+|\d+)\s*",
        '',
        text
    )

    for idx, with_block in enumerate(with_blocks):
        text = text.replace(placeholder_template.format(idx), with_block)

    text = re.sub(r'\\revert\s+\S+', '', text)

    text = re.sub(r'\\set\s+Staff\.(?!midiInstrument\b)\S+\s*=\s*[^\n\\]+', '', text)

    text = re.sub(
        r"\\(?:super|bold|italic|center-align|column|musicglyph|parentSlur|fill-line|smaller|larger)\b",
        "",
        text,
    )
    text = re.sub(r"\\(?:I|II|III|IV)[A-Za-z][\w-]*\b", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s*$", "", text)

    text = re.sub(r"<<\s*>>", "", text)
    text = re.sub(r"^[A-Za-z_][\w-]*\s*=\s*\{\s*\}", "", text, flags=re.MULTILINE)
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

    assign_re = re.compile(r"^[A-Za-z_][\w-]*\s*=")
    lines = text.splitlines()
    out = []
    depth = 0
    for line in lines:
        if assign_re.match(line) and depth > 0:
            out.extend(["}"] * depth)
            depth = 0
        out.append(line)
        line_no_comment = line.split("%", 1)[0]
        depth += line_no_comment.count("{") - line_no_comment.count("}")
        depth = max(0, depth)
    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")

    text = re.sub(r"<\s*\n\s*([+\-\d\s]+>)", r"<\1", text)
    text = re.sub(r"(\\revert\s+\w+)\s+#'", r"\1.#'", text)

    if text.startswith('version "'):
        quote_end = text.find('"', 9)
        if quote_end > 9:
            version_line = text[:quote_end + 1]
            if not version_line.rstrip().endswith('"'):
                text = version_line.rstrip() + '"' + text[quote_end:]

    text = re.sub(r'(?m)^\\language\s*$', '', text)

    text, whitelist_removed = _remove_non_whitelisted_commands(text)

    text = re.sub(r"([a-z]+[',]*\d*\.*)\s*[\^_]+(?=\s|[)\]]|$)", r'\1', text)

    text = re.sub(r'\s*[XY]-offset\s*', ' ', text)

    text = re.sub(r'([a-z]+[,\']*)\s+\.', r'\1.', text)

    text = re.sub(r' ?\{\s*\}', "", text)
    text = re.sub(r'(?:\s*-\s*)+(?=\s|$)', ' ', text)

    text = re.sub(r'\\\s+', ' ', text)

    # Remove corrupted structural lines with key/time/tempo (they're already in forma variables)
    # Pattern: lines starting with context names (PianoStaff, Staff, Voice, etc.) followed by <<
    text = re.sub(r'(?m)^\s*(?:PianoStaff|Staff|Voice|StaffGroup|ChoirStaff)\s+<<.*$', '', text)
    text = re.sub(r'(?m)^\s*Score\.[^\n]*$', '', text)
    text = re.sub(r'(?m)^\s*TupletBracket(?:\.[^\n]*)?\s*=\s*.*$', '', text)

    return text, whitelist_removed


RE_ASSIGNMENT = re.compile(r"(^|[^\w-])([A-Za-z][\w-]*)\s*=\s*", re.M)


def _variable_contains_music(rhs_content: str) -> bool:
    # Heuristic: return True if the assignment includes notes/rests or chords.
    content = rhs_content
    while True:
        match = re.search(r'\\incipit\s*\{', content)
        if not match:
            break

        brace_start = match.end() - 1  # Position of opening brace
        brace_end = grab_balanced(content, brace_start, "{", "}")

        if brace_end != -1:
            content = content[:match.start()] + content[brace_end + 1:]
        else:
            content = content[:match.start()] + content[match.end():]
            break

    note_pattern = r'\b(?:do|re|mi|fa|sol|la|si|[a-g]|r)[is|es|isbf|esbf]*[,\']*\d'
    if re.search(note_pattern, content, re.I):
        return True

    chord_pattern = r'(?<![<>])<[^<>]*\b(?:do|re|mi|fa|sol|la|si|[a-g])[is|es|isbf|esbf]*[,\']*\d?[^<>]*>(?![<>])(?:\d+)?'
    if re.search(chord_pattern, content, re.I):
        return True

    return False

def _find_all_variable_assignments(text: str) -> list[tuple[int, int, str, str]]:
    results = []
    i = 0
    n = len(text)

    while i < n:
        match = RE_ASSIGNMENT.search(text, i)
        if not match:
            break

        name = match.group(2)
        assign_start = match.start() if match.group(1) else match.start(2)

        rhs = match.end()
        while rhs < n and text[rhs].isspace():
            rhs += 1

        if rhs >= n:
            i = rhs
            continue

        # Handle different assignment types
        end = None

        # Simple markup: name = ^\markup ...
        if text[rhs] in ("_", "^"):
            m = rhs + 1
            while m < n and text[m].isspace():
                m += 1

            if text.startswith("\\markup", m):
                m += 7
                while m < n and text[m].isspace():
                    m += 1

                if m < n and text[m] == '{':
                    brace_end = grab_balanced(text, m, "{", "}")
                    end = brace_end + 1 if brace_end != -1 else m + 1
                else:
                    end = text.find('\n', m)
                    if end == -1:
                        end = n
            else:
                end = text.find('\n', rhs)
                if end == -1:
                    end = n

        # Block assignment: name = { ... }
        elif text[rhs] == '{':
            brace_end = grab_balanced(text, rhs, "{", "}")
            end = brace_end + 1 if brace_end != -1 else rhs + 1

        # \relative, \transpose, etc.
        elif text.startswith("\\relative", rhs) or text.startswith("\\transpose", rhs):
            brace_pos = text.find('{', rhs)
            if brace_pos != -1:
                brace_end = grab_balanced(text, brace_pos, "{", "}")
                end = brace_end + 1 if brace_end != -1 else brace_pos + 1
            else:
                end = text.find('\n', rhs)
                if end == -1:
                    end = n

        # Angle brackets: name = << ... >>
        elif text.startswith("<<", rhs):
            angle_end = grab_angles(text, rhs)
            end = angle_end if angle_end != -1 else rhs + 2

        if end is not None:
            results.append((assign_start, end, name, text[assign_start:end]))
            i = end
        else:
            i = rhs + 1

    return results


def _remove_engraving_only_variables(text: str) -> tuple[str, int]:
    assignments = _find_all_variable_assignments(text)
    if not assignments:
        return text, 0

    removed_count = 0

    for assign_start, assign_end, name, full_text in reversed(assignments):
        if not _variable_contains_music(full_text):
            line_start = assign_start
            while line_start > 0 and text[line_start - 1] not in '\n':
                line_start -= 1

            end_pos = assign_end
            if end_pos < len(text) and text[end_pos] == '\n':
                end_pos += 1

            text = text[:line_start] + text[end_pos:]
            removed_count += 1

    return text, removed_count


def _remove_engraving_only_paragraphs(text: str) -> tuple[str, int]:
    assignments = _find_all_variable_assignments(text)
    assignment_ranges = [(start, end) for start, end, _, _ in assignments]

    def is_inside_assignment(pos):
        # Check if position is inside a variable assignment.
        for start, end in assignment_ranges:
            if start <= pos < end:
                return True
        return False

    paragraphs = re.split(r'(\n\s*\n)', text)

    result_parts = []
    removed_count = 0
    current_pos = 0

    for i, para in enumerate(paragraphs):
        para_start = current_pos
        para_end = current_pos + len(para)
        current_pos = para_end

        if re.match(r'^\n\s*\n$', para):
            if result_parts and i < len(paragraphs) - 1:
                result_parts.append('\n\n')
            continue

        if not para.strip():
            continue

        if is_inside_assignment(para_start):
            result_parts.append(para)
            continue

        if _variable_contains_music(para):
            result_parts.append(para)
        else:
            if re.search(r"\\language\b", para, re.I):
                result_parts.append(para)
                continue
            lines = para.strip().split('\n')
            all_comments = all(line.strip().startswith('%') or not line.strip() for line in lines)

            if all_comments and any(line.strip() for line in lines):
                result_parts.append(para)
            else:
                removed_count += 1

    return '\n\n'.join(result_parts), removed_count


def _remove_engraving_only_top_level(text: str) -> tuple[str, int]:
    assignments = _find_all_variable_assignments(text)
    assignment_ranges = [(start, end) for start, end, _, _ in assignments]

    def is_inside_assignment(pos):
        # Check if position is inside a variable assignment.
        for start, end in assignment_ranges:
            if start <= pos < end:
                return True
        return False

    lines = text.split('\n')
    result_lines = []
    removed_count = 0
    current_pos = 0

    for line in lines:
        line_start = current_pos
        line_end = current_pos + len(line) + 1  # +1 for newline
        current_pos = line_end

        if not line.strip():
            result_lines.append(line)
            continue

        if is_inside_assignment(line_start):
            result_lines.append(line)
            continue

        if re.search(r'\\version\b|\\language\b', line, re.I):
            result_lines.append(line)
            continue

        if line.strip().startswith('%'):
            result_lines.append(line)
            continue

        if _variable_contains_music(line):
            result_lines.append(line)
        else:
            removed_count += 1

    return '\n'.join(result_lines), removed_count


try:
    from lilynorm.utils.options import NormOptions
except Exception:
    class NormOptions:  # type: ignore[override]
        keep_engraving: bool = True  # default: keep engravings


def run(text: str, opts: NormOptions) -> str:
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

        cleaned, metadata_count = _remove_metadata_headers(cleaned)
        _add_count("metadata", metadata_count)
        cleaned = _split_inline_assignments(cleaned)

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
