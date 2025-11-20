\version "2.24.4"
\language "italiano"

\score {
  {
    \key sol \major
    R4*2
    sol''8 la''8 sol''8 re''8
    re''4 re''4
    sol''8 la''8 sol''8 re''8
    re''4 re''4
    \repeat volta 2 {
      fad''8 mi''8 fad''8 sol''8
      sol''8 la''8 sol''8 re''8
    }
    r4 r4 r2
    \bar "|."
  }
}