"""
Simple, practical metrics for evaluating generated LilyPond music.

Focus on thesis-defensible metrics:
1. Syntax validity (can LilyPond compile it?)
2. Basic statistics (length, pitch range, intervals)
3. Simple music metrics (repetition, note distribution)

No 500-line monster. Keep it practical.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


@dataclass
class MusicStats:
    """Statistics about a LilyPond piece."""
    total_notes: int = 0
    unique_pitches: int = 0
    pitch_range: Tuple[str, str] = ("", "")
    avg_interval: float = 0.0
    repetition_ratio: float = 0.0
    has_valid_syntax: bool = False
    note_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.note_distribution is None:
            self.note_distribution = {}


class LilyPondMetrics:
    """Simple metrics for LilyPond music evaluation.
    
    Just enough for your thesis defense:
    - Can LilyPond compile it? (critical!)
    - How long is it?
    - What's the pitch range?
    - How repetitive is it?
    """
    
    # Note names in Italian (solfège) and English
    NOTE_PATTERN = re.compile(
        r"\b(do|re|mi|fa|sol|la|si|[a-g])[',]*(?:is|es|s|b)?\d*",
        re.IGNORECASE
    )
    
    # Italian to English pitch mapping
    ITALIAN_TO_ENGLISH = {
        "do": "c", "re": "d", "mi": "e", "fa": "f",
        "sol": "g", "la": "a", "si": "b"
    }
    
    # Pitch ordering for range calculation
    PITCH_ORDER = ["c", "d", "e", "f", "g", "a", "b"]
    
    def __init__(self, lilypond_command: str = "lilypond"):
        """Initialize metrics.
        
        Args:
            lilypond_command: Command to run LilyPond (e.g., "lilypond" or full path)
        """
        self.lilypond_command = lilypond_command
    
    def check_syntax_validity(self, lilypond_text: str) -> bool:
        """Check if LilyPond can compile the text without errors.
        
        This is THE most important metric: does it actually work?
        
        Args:
            lilypond_text: LilyPond source code
            
        Returns:
            True if compilation succeeds, False otherwise
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ly",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(lilypond_text)
                temp_path = f.name
            
            # Try to compile (output to /dev/null equivalent)
            result = subprocess.run(
                [self.lilypond_command, "-o", tempfile.gettempdir(), temp_path],
                capture_output=True,
                timeout=30,
            )
            
            Path(temp_path).unlink(missing_ok=True)
            
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Syntax check failed: {e}")
            return False
    
    def extract_notes(self, lilypond_text: str) -> List[str]:
        """Extract all note names from LilyPond text.
        
        Args:
            lilypond_text: LilyPond source code
            
        Returns:
            List of note names (normalized to English)
        """
        notes = []
        for match in self.NOTE_PATTERN.finditer(lilypond_text):
            note = match.group(1).lower()
            # Convert Italian to English
            note = self.ITALIAN_TO_ENGLISH.get(note, note)
            # Remove octave markers and accidentals for base pitch
            base_note = note[0] if note else ""
            if base_note in self.PITCH_ORDER:
                notes.append(base_note)
        
        return notes
    
    def calculate_pitch_range(self, notes: List[str]) -> Tuple[str, str]:
        """Calculate pitch range (lowest to highest).
        
        Args:
            notes: List of note names
            
        Returns:
            Tuple of (lowest_note, highest_note)
        """
        if not notes:
            return ("", "")
        
        # Sort by pitch order
        sorted_notes = sorted(set(notes), key=lambda n: self.PITCH_ORDER.index(n))
        return (sorted_notes[0], sorted_notes[-1])
    
    def calculate_repetition_ratio(self, notes: List[str]) -> float:
        """Calculate how repetitive the music is.
        
        Simple metric: ratio of most common note to total notes.
        
        Args:
            notes: List of note names
            
        Returns:
            Repetition ratio (0.0 to 1.0)
        """
        if not notes:
            return 0.0
        
        counter = Counter(notes)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(notes)
    
    def calculate_avg_interval(self, notes: List[str]) -> float:
        """Calculate average interval size.
        
        Simple approximation using pitch order.
        
        Args:
            notes: List of note names
            
        Returns:
            Average interval (in scale steps)
        """
        if len(notes) < 2:
            return 0.0
        
        intervals = []
        for i in range(len(notes) - 1):
            try:
                idx1 = self.PITCH_ORDER.index(notes[i])
                idx2 = self.PITCH_ORDER.index(notes[i + 1])
                intervals.append(abs(idx2 - idx1))
            except ValueError:
                continue
        
        return sum(intervals) / len(intervals) if intervals else 0.0
    
    def evaluate(self, lilypond_text: str) -> MusicStats:
        """Evaluate a LilyPond piece.
        
        Complete evaluation with all metrics.
        
        Args:
            lilypond_text: LilyPond source code
            
        Returns:
            MusicStats with all calculated metrics
        """
        stats = MusicStats()
        
        # Extract notes
        notes = self.extract_notes(lilypond_text)
        
        # Basic counts
        stats.total_notes = len(notes)
        stats.unique_pitches = len(set(notes))
        
        # Note distribution
        stats.note_distribution = dict(Counter(notes))
        
        # Pitch range
        stats.pitch_range = self.calculate_pitch_range(notes)
        
        # Intervals
        stats.avg_interval = self.calculate_avg_interval(notes)
        
        # Repetition
        stats.repetition_ratio = self.calculate_repetition_ratio(notes)
        
        # Syntax validity (most important!)
        stats.has_valid_syntax = self.check_syntax_validity(lilypond_text)
        
        return stats
    
    def evaluate_batch(self, lilypond_texts: List[str]) -> List[MusicStats]:
        """Evaluate multiple pieces.
        
        Args:
            lilypond_texts: List of LilyPond source codes
            
        Returns:
            List of MusicStats
        """
        return [self.evaluate(text) for text in lilypond_texts]


def evaluate_generated_files(
    file_paths: List[str | Path],
    lilypond_command: str = "lilypond",
) -> Dict[str, Any]:
    """Evaluate multiple generated .ly files and summarize results.
    
    Perfect for thesis: "I generated 50 pieces, 42 compiled successfully..."
    
    Args:
        file_paths: List of paths to .ly files
        lilypond_command: LilyPond command
        
    Returns:
        Dictionary with summary statistics
    """
    metrics = LilyPondMetrics(lilypond_command=lilypond_command)
    
    all_stats = []
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        stats = metrics.evaluate(text)
        all_stats.append(stats)
    
    # Summarize
    if not all_stats:
        return {}
    
    valid_count = sum(1 for s in all_stats if s.has_valid_syntax)
    total_count = len(all_stats)
    
    summary = {
        "total_files": total_count,
        "valid_syntax_count": valid_count,
        "valid_syntax_rate": valid_count / total_count if total_count > 0 else 0.0,
        "avg_total_notes": sum(s.total_notes for s in all_stats) / total_count,
        "avg_unique_pitches": sum(s.unique_pitches for s in all_stats) / total_count,
        "avg_repetition_ratio": sum(s.repetition_ratio for s in all_stats) / total_count,
        "avg_interval": sum(s.avg_interval for s in all_stats) / total_count,
        "all_stats": all_stats,
    }
    
    return summary


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
    """Print evaluation summary in a nice format.
    
    Args:
        summary: Summary dictionary from evaluate_generated_files
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total files evaluated: {summary['total_files']}")
    print(f"Valid syntax (compilable): {summary['valid_syntax_count']} "
          f"({summary['valid_syntax_rate']*100:.1f}%)")
    print(f"\nAverage statistics:")
    print(f"  - Total notes: {summary['avg_total_notes']:.1f}")
    print(f"  - Unique pitches: {summary['avg_unique_pitches']:.1f}")
    print(f"  - Repetition ratio: {summary['avg_repetition_ratio']:.2f}")
    print(f"  - Average interval: {summary['avg_interval']:.2f} steps")
    print("="*60 + "\n")


# Simple CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LilyPond files")
    parser.add_argument(
        "files",
        nargs="+",
        help=".ly files to evaluate",
    )
    parser.add_argument(
        "--lilypond",
        default="lilypond",
        help="LilyPond command",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Evaluate
    print(f"Evaluating {len(args.files)} files...")
    summary = evaluate_generated_files(args.files, args.lilypond)
    
    # Print results
    print_evaluation_summary(summary)
    
    # Print individual results
    if summary.get("all_stats"):
        print("\nIndividual file results:")
        for i, (path, stats) in enumerate(zip(args.files, summary["all_stats"])):
            status = "✓" if stats.has_valid_syntax else "✗"
            print(f"{status} {Path(path).name}: "
                  f"{stats.total_notes} notes, "
                  f"range {stats.pitch_range[0]}-{stats.pitch_range[1]}, "
                  f"rep={stats.repetition_ratio:.2f}")
