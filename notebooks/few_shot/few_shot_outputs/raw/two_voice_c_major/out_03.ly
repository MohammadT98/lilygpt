\version "2.24.4"
\language "italiano"

\score {
  \key do \major
  \time 4/4
  <<
    {
      R4
      \p
      do'4. re'8 mi'8. re'16 \slurUp
      \staccato
      fa'8. la'16 \slurDown
      \fermata
      si'4. \trill
      re''8 \accent re''8 mi''8. re''16
      fad''8 sol''8 fa''8. sol''16
      \cresc \tempo 4=120
      \repeat volta 2 {
        sol''8 si''8 fad''8 sol''8
        sol''4 sol''4
      }
    }
    \\
    {
      \repeat volta 2 {
        do'8 re'8 mi'8 fa'8
        sol'8 la'8 si'8 do''8
        \arpeggio
        do'8 re'8 mi'8 fa'8
        sol'8 la'8 si'8 do''8
      }
      \break
      \tempo 4=90
      fad'8 re'8 do'8 re'8
      fad'8 re'8 fad'8 re'8
      \fermata
    }
  >>
}