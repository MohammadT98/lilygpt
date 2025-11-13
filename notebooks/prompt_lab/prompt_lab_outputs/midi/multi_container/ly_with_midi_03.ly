\version "2.24.4"


\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
c4 d4 e4 f4 | g4 a4 b4 c4 | d4 e4 f4 g4 | a4 b4 c4 d4 | e4 f4 g4 a4 | b4 c4 d4 e4 | f4 g4 a4 b4 | c4 d4 e4 f4 |
}
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
c4 e4 g4 b4 | a4 f4 d4 c4 | g4 e4 c4 a4 | b4 d4 f4 e4 | c4 e4 g4 b4 | a4 f4 d4 c4 | g4 e4 c4 a4 | b4 d4 f4 e4 |
}
  }
  \layout {}
  \midi {}
}
