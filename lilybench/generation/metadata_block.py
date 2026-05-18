"""Render the ``%% === METADATA ===`` block used in every generation prompt.

Each prompt in the bank carries a metadata header so the four backbones
condition on identical composer/period/form/ensemble/part fields. The
block format matches the training distribution of BMdataset so future
adapter-tuned variants can reuse the same prompts byte-for-byte.
"""

from __future__ import annotations

from typing import Any, Mapping

_FIELDS = ("composer", "period", "musical_form", "ensemble", "part")


def render_metadata_block(metadata: Mapping[str, Any] | None) -> str:
    """Return a ``%% === METADATA === / ... / %% === END METADATA ===`` block.

    Empty / ``None`` fields are omitted. List-valued fields are joined with
    ``", "`` so the rendered form matches the BMdataset preprocessing.
    """
    lines = ["%% === METADATA ==="]
    if metadata:
        for key in _FIELDS:
            val = metadata.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (list, tuple)):
                if not val:
                    continue
                val = ", ".join(str(v) for v in val)
            lines.append(f"%% {key}: {val}")
    lines.append("%% === END METADATA ===")
    return "\n".join(lines) + "\n"
