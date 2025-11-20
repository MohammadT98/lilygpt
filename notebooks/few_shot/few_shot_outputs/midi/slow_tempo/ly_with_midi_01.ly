\version "2.24.4"

\language "italiano"

\score {
  \tempo 4 = 60
  {
    R1*2
    \repeat volta 2 {
      re''8 mi''16 fa''16 sol''8 la''8
      sol''8 fa''8 mi''8 re''8
      re''4 mi''4
    }
    \alternative {
      { re''8 mi''16 fa''16 sol''8 la''8 }
      { re''8 mi''16 fa''16 sol''8 la''8 }
    }
    \repeat volta 2 {
      la''8 sol''8 fa''8 mi''8
      re''8 mi''8 fa''8 sol''8
      re''4 mi''4
    }
    r2
  }

  \layout {}

  \midi {}
}
