\version "2.24.4"

\score {
  \new Staff {
\def \phraseA { c4 d4 e4 f4 }
\relative c' {
  \key c \major
  \time 4/4
  \tempo 4 = 100
  \phraseA | \phraseA | \phraseA | \phraseA | \phraseA | \phraseA | \phraseA | \phraseA |
}
}
  \layout {}
  \midi {}
}
