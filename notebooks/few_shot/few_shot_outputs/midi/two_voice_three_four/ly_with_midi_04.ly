\version "2.24.4"

\language "italiano"

\score {
  \time 3/4
  <<
    { \p
      R4 R4 R4
      re'8[ mi'16 fa'16] re'4
      mi'8[ si'16 la'16] mi'4
      sol'8[ re'16 mi'16] sol'4
      re'8[ la'16 re'16] re'4
      \fermata
    }
    { \mf
      R4 R4 R4
      fad''8[ sol''16 la''16] fad''4
      la''8[ sol''16 re''16] la''4
      sol''8[ re''16 mi''16] sol''4
      fad''!8[ sol''16 la''16] fad''4
      \bar "|."
    }
  >>

  \layout {}

  \midi {}
}
