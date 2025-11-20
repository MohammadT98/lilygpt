\version "2.24.4"

\language "italiano"

\score {
  {
    \tempo 4 = 160
    R1
    sol''8 do''8 la'4 mi'8 re'8
    sol''8 ( si''16 la''16 sol''8 ) re''4
    [ sol'' mi'' do'' ]
    re''8 mi''16 re''16 do''8 si'8
    re''8 la'8 do''8 si''8
    \repeat volta 2 {
      mi''8 la''8 re''8 do''8
      sol''8 si''16 la''16 sol''8 re''4
    }
    \alternative { { sol''4 r4 } { sol''4 r4 } }
  }

  \layout {}

  \midi {}
}
