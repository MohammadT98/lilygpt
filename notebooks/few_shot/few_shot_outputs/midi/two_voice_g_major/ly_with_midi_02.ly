\version "2.24.4"

\language "italiano"

\score {
  \key sol \major
  \time 4/4
  <<
    { \voiceOne
      \tempo 4 = 120
      ( sol''4 re''8 mi''8. fa''16 sol''8 ) r4
      la''2 r4 r8 r8
      sol''4. si''8 la''4 r4
      re''4 mi''8. fa''16 sol''4 r4
      ( sol''4 re''8 mi''8. fa''16 sol''8 ) r4
      la''2 r4 r8 r8
      sol''4. si''8 la''4 r4
      re''4 mi''8. fa''16 sol''4 r4
    }
    \\ 
    { \voiceTwo
      \p \grace{re,} sol,4 re,4 mi,4 fa,4
      \p la,2 sol,4 r4
      \p ( sol,4 re,4 mi,4 fa,4 )
      \p la,2 sol,4 r4
      \f ( sol,4 re,4 mi,4 fa,4 )
      \f la,2 sol,4 r4
      \f ( sol,4 re,4 mi,4 fa,4 )
      \f la,2 sol,4 r4
    }
  >>

  \layout {}

  \midi {}
}
