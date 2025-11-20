\version "2.24.4"
\language "italiano"

\score {
  <<
    \new Staff \with { \clef treble } {
      \key sol \major
      R2.*4
      sol''4 la''8 sol''8 si''8 sol''8 la''8 sol''8
      r2
      re''8 re''8 re''8 re''8 si''8 re''8 si''8 re''8
      r2
      la''16 sol''16 fa''16 la''16 sol''16 la''16 sol''16 fa''16
      r2
      re''8 la''8 re''8 la''8 re''8 la''8 re''8 la''8
      r2
    }
  >>
}