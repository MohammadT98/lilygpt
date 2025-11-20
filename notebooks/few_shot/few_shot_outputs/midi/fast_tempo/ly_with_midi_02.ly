\version "2.24.4"

\language "italiano"

\tempo 4 = 160

\score {
  {
    R2.*2

    do''8 re'' mi'' fa'' sol'' la'' si'' sol'' |
    do''8 re'' mi'' fa'' sol'' la'' si'' sol'' |
    do''8 re'' mi'' fa'' sol'' la'' si'' sol'' |
    R4 r4

    \repeat volta 2 {
      sol''8 la'' si'' do''' la'' sol'' fa'' mi'' |
      sol''8 la'' si'' do''' la'' sol'' fa'' mi'' |
      sol''8 la'' si'' do''' la'' sol'' fa'' mi'' |
      sol''4 r4
    }

    \bar "|."
  }

  \layout {}

  \midi {}
}
