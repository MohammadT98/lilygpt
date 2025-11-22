\version "2.24.4"

\language "italiano"

\score {
  \key sol \major
  \time 4/4
  <<
    {
      \p
      R1
      r4 sol''8. la''16 sol''8 la''8. sol''16
      do''4 re''8 mi''8 fa''8 re''8
      si''8[ si''16 la''16 sol''8. la''16 ]
      sol''4 ~ sol''8 fa''8. fa''16
      fa''8[ mi''16 re''16 mi''8. mi''16 ]
      la''16 sol''16 fa''16 mi''16
      re''8 sol''16 fa''16
      mi''8[ mi''8 la''8. mi''16 ]
      sol''8 la''16 sol''16 sol''8. la''16
      sol''4 r4 r2
    }
    \\ {
      \p
      R1
      r4 sol2 ~ sol2
      do2 re2
      mi2 fa2
      sol2 la2
      do'4 re'8 mi'8 fa'8 re'8
      si'8[ si'16 la'16 sol'8. la'16 ]
      sol'4 ~ sol'8 fa'8. fa'16
      fa'8[ mi'16 re'16 mi'8. mi'16 ]
      la'16 sol'16 fa'16 mi'16
      re'8 sol'16 fa'16
      mi'8[ mi'8 la'8. mi'16 ]
      sol'8 la'16 sol'16 sol'8. la'16
      sol'4 r4 r2
    }
  >>

  \layout {}

  \midi {}
}
