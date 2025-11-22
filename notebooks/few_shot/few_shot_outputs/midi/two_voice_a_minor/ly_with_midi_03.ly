\version "2.24.4"

\language "italiano"

\score {
  <<
    {
      \key la \minor
      \time 4/4
      r4
      la'8[ mi'16 re'16] fa'8 si'4
      re'8[ si'16 la'16] re''8 r8
      la'4 ~ la'8 r8
      fa'8[ mi'16 re'16] la'8 ~ la'8
      re''8[ re''16 re''16] mi'8 r8
      \grace { re''8 } la''8~ la''4 \p
      sol'8[ la'16 sol'16] re'8 r8
      mi'4 \f
    }
    \\
    {
      \key la \minor
      \time 4/4
      r2
      do'8[ re'8] mi'8[ fa'8] sol'8~ sol'8
      re'8[ sol'8] mi'8[ re'8] do'8~ do'8
      la'4 r4
      re'8[ mi'8] fa'8[ si'8] mi'8~ mi'8
      re'4 r4
      \grace { re'8 } la'8~ la'4 \p
      fa'8[ mi'16 re'16] sol'8 r8
      do'4 \f
    }
  >>

  \layout {}

  \midi {}
}
