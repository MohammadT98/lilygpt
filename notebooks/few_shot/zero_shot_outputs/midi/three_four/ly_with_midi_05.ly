\version "2.24.4"

\language "italiano"

\score {
  <<
    \new Staff <<
      \time 3/4
      \tempo "Allegro moderato" 4=120
      \repeat volta 2 {
        r4 do'8 re'8 mi'8 |
        fa'4 sol'8 la'8 si'8 |
        re'4 mi'8 fa'8 sol'8 |
        la'4 si'8 do''8 re''8 |
        mi''4 re''8 la'8 sol'8 |
        fa'4 mi'8 re'8 do'8 |
        si'4 la'8 sol'8 fa'8 |
        sol'4 fa'8 re'8 mi'8 |
      }
      \alternative {
        { r4 r8 r8 r8 | \bar "|." } 
        { r4 do''8 re''8 mi''8 | \bar "|." } 
      }
    >>
  >>

  \layout {}

  \midi {}
}
