\version "2.24.4"


\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
c4 \grace {c8} d4 e4 f4 | \tuplet 3/2 { g4 a4 b4 } c4 d4 | \times 2/3 { e4 f4 g4 a4 b4 c4 } | \acciaccatura c8 d4 e4 f4 g4 | \appoggiatura a8 b4 c4 d4 e4 | \ottava 1 { f4 g4 a4 b4 } \ottava 0 | e4 f4 g4 a4 | c4 d4 e4 f4 |
}
  }
  \layout {}
  \midi {}
}
