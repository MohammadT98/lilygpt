\version "2.24.4"

\language "italiano"

\score {
  <<
    \new Voice = "melody" {
      \key sol \major
      \time 4/4
      r4 sol'8 la'8 |
      mi'8 fa'8 sol'8 la'8 |
      re'8 mi'8 fa'8 sol'8 |
      la'8 si'8 do''8 re''8 |
      r4 sol'8 si'8 |
      mi''8 re''8 la''8 sol''8 |
      re''8 mi''8 la''8 sol''8 |
      r2. r2
    }
  >>

  \layout {}

  \midi {}
}
