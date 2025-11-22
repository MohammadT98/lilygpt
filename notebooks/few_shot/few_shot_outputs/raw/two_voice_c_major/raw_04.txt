\version "2.24.4"
\language "italiano"

\score {
  <<
    { \voiceOne
      \key do \major
      \time 4/4
      \tempo "Allegro"
      \p
      do''8-. re''8-. mi''8-. fa''8-. \fermata |
      sol''4 ~ sol''8 la''8 \fermata |
      fad''8 re''8 | sol''8 la''8 | do''4 r4 r4 |
      re''8 mi''16 re''16 mi''8. mi''16 | sol''4 r4 r4 |
      la''8 la''8 la''8 la''8 | do''8 mi''8 re''8 fa''8 }
    \\ { \voiceTwo
      \key do \major
      \time 4/4
      \tempo "Allegro"
      \mf
      do'8-. re'8-. mi'8-. fa'8-. \fermata |
      sol'4 ~ sol'8 la'8 \fermata |
      fad'8 re'8 | sol'8 la'8 | do'4 r4 r4 |
      re'8 mi'16 re'16 mi'8. mi'16 | sol'4 r4 r4 |
      la'8 la'8 la'8 la'8 | do'8 mi'8 re'8 fa'8 }
  >>
}