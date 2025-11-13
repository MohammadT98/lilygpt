\version "2.24.4"

\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\seq1 = { c4 d4 e4 f4 }
\seq2 = { g4 a4 b4 c'4 }
\seq1 | \seq1 | \seq1 | \seq1 | \seq2 | \seq2 | \seq2 | \seq2 |
}
}
  \layout {}
  \midi {}
}
