\version "2.24.4"

\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\barSeq = { c4 d4 e4 f4 }
\barSeq | \barSeq | \barSeq | \barSeq | \barSeq | \barSeq | \barSeq | \barSeq |
}
}
  \layout {}
  \midi {}
}
