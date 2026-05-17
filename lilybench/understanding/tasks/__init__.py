"""Built-in understanding tasks shipped with the paper.

Importing this subpackage registers all ten tasks.
"""

from __future__ import annotations

from lilybench.understanding.tasks import (
    bar_count,
    bar_sequencing,
    composer_recognition,
    emotion_recognition,
    error_detection,
    genre_recognition,
    metadata_prediction,
    metadata_qa,
    music_captioning,
    next_bar_prediction,
)


# Marker symbol used by the parent package to confirm registration ran.
_ensure_registered = True
