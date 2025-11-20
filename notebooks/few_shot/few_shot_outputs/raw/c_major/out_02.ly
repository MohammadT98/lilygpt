\version "2.24.4"
\language "italiano"

\score {
  <<
    \new Staff {
      \key do \major
      \time 4/4
      R1*2
      (do'' re'' mi'' fa'' sol'' la'' si'' do'''8)
      R4
      <la'' si'' do''' re'''>4
      R2
      do''8 mi''8 fa''8 sol''8 la''8 re''8 mi''8 fa''8
      R1*3
      re''8 re''8 re''8 re''8 re''8 re''8 re''8 re''8
      r4 r8 re''8
      mi''8 fa''8 sol''8 la''8
      R1
    }
  >>
}