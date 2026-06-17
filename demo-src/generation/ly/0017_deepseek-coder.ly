\language "nederlands"
\version "2.24.0"

\header {
  composer = "Boismortier"
  period = "Late Baroque"
  musical_form = "suite"
  ensemble = "violin, bassoon, contrabass, oboe, flute, viola"
  part = "dessus"
}

dessus = {
  \time 4/4
  \key c \major

  % Movement 1
  c'4 d' e' f' | g' a' b' c'' | d'' e'' f'' g'' | a'' b'' c'' d'' |
  e'' f'' g'' a'' | b'' c'' d'' e'' | f'' g'' a'' b'' | c''1 \bar "|."

  % Movement 2
  e'4 f' g' a' | b' c'' d'' e'' | f'' g'' a'' b'' | c''1 |
  c''4 d'' e'' f'' | g'' a'' b'' c'' | d'' e'' f'' g'' | a''1 \bar "|."

  % Movement 3
  c'4 d' e' f' | g' a' b' c'' | d'' e'' f'' g'' | a'' b'' c'' d'' |
  e'' f'' g'' a'' | b'' c'' d'' e'' | f'' g'' a'' b'' | c''1 \bar "|."

  % Movement 4
  e'4 f' g' a' | b' c'' d'' e'' | f'' g'' a'' b'' | c''1 |
  c''4 d'' e'' f'' | g'' a'' b'' c'' | d'' e'' f'' g'' | a''1 \bar "|."
}

\score {
  \new StaffGroup <<
    \new Staff \with { instrumentName = "Violin" } { \dessus }
    \new Staff \with { instrumentName = "Bassoon" } { \dessus }
    \new Staff \with { instrumentName = "Contrabass" } { \dessus }
    \new Staff \with { instrumentName = "Oboe" } { \dessus }
    \new Staff \with { instrumentName = "Flute" } { \dessus }
    \new Staff \with { instrumentName = "Viola" } { \dessus }
  >>
  \layout { }
  \midi { }
}
