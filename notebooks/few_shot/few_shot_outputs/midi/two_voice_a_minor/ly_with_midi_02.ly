\version "2.24.4"

\language "italiano"

\score {
  <<
    { \key la \minor \time 4/4
      R1
      > la''4 mi''4 re''4 la''4
      > re''4 mi''4 la''4 re''4
      R2*2
      la''8[ mi''8 re''8. la''16 ]\fermata R2
      sol''4 fa''4 sol''8 fa''8 R4
      la''4 mi''4 re''4 la''4
      R1
    }
    \\ { \key la \minor \time 4/4
      R4 r4 r4 r4
      la''8 la''8 la''8 la''8 la''8 la''8 la''8 la''8
      la''4 la''4 la''4 la''4
      R2*2
      la''8[ la''8 la''8. la''16 ]\fermata R2
      la''4 la''4 la''8 la''8 R4
      la''4 la''4 la''4 la''4
      R1
    }
  >>

  \layout {}

  \midi {}
}
