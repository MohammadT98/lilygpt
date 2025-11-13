\version "2.24.4"

\score {
  \new Staff {
\relative c' { \key c \major \time 4/4 \tempo 4 = 100 c4 d4 e4 f4 | g4 a4 b4 c5 | d5 e5 f5 g5 | a5 b5 c6 d6 | \transpose c f { c4 d4 e4 f4 | g4 a4 b4 c5 | d5 e5 f5 g5 | a5 b5 c6 d6 | } }
}
  \layout {}
  \midi {}
}
