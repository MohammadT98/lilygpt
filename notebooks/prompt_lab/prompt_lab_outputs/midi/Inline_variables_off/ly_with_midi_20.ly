\version "2.24.4"

\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\foo = { c4 d4 e4 f4 }
\foo | \foo | \foo | \foo | \foo | \foo | \foo | \foo |
}
}
  \layout {}
  \midi {}
}
