\version "2.24.2"
\language "nederlands"

\header {
  composer = "Boismortier"
  title = "Suite"
}

\relative c' {
  \clef treble
  \key c \major
  \time 4/4

  % Allemande
  \tempo "Lento"
  c4 d e f | g a b c | d e f g | a b c d |

  % Courante
  \tempo "Andante"
  e4 f g a | b c d e | f g a b | c d e f |

  % Sarabande
  \tempo "Adagio"
  g4 a b c | d e f g | a b c d | e f g a |

  % Gigue
  \tempo "Allegro"
  c4 d e f | g a b c | d e f g | a b c d |
}
