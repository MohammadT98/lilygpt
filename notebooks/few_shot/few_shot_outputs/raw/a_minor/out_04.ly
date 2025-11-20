\version "2.24.4"
\language "italiano"

\score {
  {
    \key la \minor
    R1
    la'4 \p re'8 la'8 [re'8 la'8]
    re'4 \f re'8 mi'8 la'8 re'8
    si8 la8 sol8 fa8 mi8
    la'8 la'8 la'8 la'8
    re'8 re'8 re'8 re'8
    mi'8 si'8 la'8 re'8
    sol'8 fa'8 mi'8 re'8
    \repeat volta 2 {
      la'4 r4
      re'4 r4
    }
    \alternative {
      { r2. }
      { r4 }
    }
  }
}