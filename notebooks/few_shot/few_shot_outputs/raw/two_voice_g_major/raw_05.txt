\version "2.24.4"
\language "italiano"

\score {
  <<
    { \key sol \major \time 4/4
      (sol''4. la''8) \p
      fa''16 sol''16 sol''8. la''16 \> \sustain
      sol''8 ~ sol''8 re''8 mi''8. mi''16 \f
      sol''2
      re''4. mi''8 \p
      fa''4 sol''8. la''16 \! \sustain
      la''4 la''4 \pp
      sol''2 \f
    }
    { \key sol \major \time 4/4
      sol''4 sol''4 \pp
      re''4 mi''4 \p
      sol''4 la''4 \p
      sol''2 \p
      re''4 mi''4 \pp
      sol''4 fa''4 \p
      sol''4 sol''4 \p
      sol''2 \f
    }
  >>
}