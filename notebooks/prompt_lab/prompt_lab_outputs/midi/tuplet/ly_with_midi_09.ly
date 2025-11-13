\version "2.24.4"


\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
c4 d4 e4 f4 | \tuplet 3/2 { g8 a8 b8 } c4 d4 e4 | \tuplet 3/2 { f8 g8 a8 } b4 c4 d4 | e4 f4 g4 a4 | \tuplet 3/2 { b8 c8 d8 } e4 f4 g4 | a4 b4 c4 d4 | \tuplet 3/2 { e8 f8 g8 } a4 b4 c4 | d4 e4 f4 g4 |
}
  }
  \layout {}
  \midi {}
}
