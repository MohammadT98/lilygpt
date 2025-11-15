\version "2.24.4"

\score {
  \new Staff {
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
\transpose c c' {
  c4 d4 e4 f4 | g4 a4 b4 c'4 | d'4 e'4 f'4 g'4 | a'4 b'4 c''4 d''4 |
}
\transpose c c' {
  e4 f4 g4 a4 | b4 c'4 d'4 e'4 | f'4 g'4 a'4 b'4 | c''4 d''4 e''4 f''4 |
}
}
}
  \layout {}
  \midi {}
}
