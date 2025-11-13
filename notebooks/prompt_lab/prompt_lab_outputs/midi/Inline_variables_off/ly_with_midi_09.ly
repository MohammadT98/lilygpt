\version "2.24.4"

\score {
  \new Staff {
\first = { c4 d4 e4 f4 }
\second = { g4 a4 b4 c4 }
\third = { d4 e4 f4 g4 }
\fourth = { a4 b4 c4 d4 }
\fifth = { e4 f4 g4 a4 }
\sixth = { b4 c4 d4 e4 }
\seventh = { f4 g4 a4 b4 }
\eighth = { c4 d4 e4 f4 }
\relative c' {
  \key c \major \time 4/4 \tempo 4 = 100
  \first | \second | \third | \fourth | \fifth | \sixth | \seventh | \eighth |
}
}
  \layout {}
  \midi {}
}
