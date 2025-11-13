\version "2.24.4"

\score {
  \new Staff {
\myBarA = { c4 d4 e4 f4 }
\myBarB = { g4 a4 b4 c4 }
\myBarC = { d4 e4 f4 g4 }
\myBarD = { a4 b4 c4 d4 }
\myBarE = { e4 f4 g4 a4 }
\myBarF = { b4 c4 d4 e4 }
\myBarG = { f4 g4 a4 b4 }
\myBarH = { c4 d4 e4 f4 }
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\myBarA | \myBarB | \myBarC | \myBarD | \myBarE | \myBarF | \myBarG | \myBarH |
}
}
  \layout {}
  \midi {}
}
