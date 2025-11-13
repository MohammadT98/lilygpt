\version "2.24.4"

mySeq = { c4 d e f }

\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\mySeq | \mySeq | \mySeq | \mySeq | \mySeq | \mySeq | \mySeq | \mySeq |
}
}
  \layout {}
  \midi {}
}
