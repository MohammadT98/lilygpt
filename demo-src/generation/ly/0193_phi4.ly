\version "2.22.0"
\language "english"

\header {
  composer = "de Lalande"
  title = "Sinfonia"
  instrument = "Basso"
}

\score {
  \new Staff \with {
    instrumentName = "Basso"
  } {
    \relative c' {
      \key g \minor
      \time 4/4

      % Baroque style basso continuo
      d2 g,4 a
      b2 c4 d
      e2 f4 g
      a2 b4 c
      d2 e4 f
      g2 a4 b
      c2 d4 e
      f2 g4 a

      % Repeat and develop the motif
      b2 a4 g
      f2 e4 d
      c2 b4 a
      g2 f4 e
      d2 c4 b
      a2 g4 f
      e2 d4 c
      b2 a4 g

      % Cadence
      c2 b4 a
      g2 f4 e
      d1
    }
  }
}
