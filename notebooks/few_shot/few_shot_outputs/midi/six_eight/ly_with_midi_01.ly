\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \clef treble
    \time 6/8
    \key do \major
    \repeat volta 2 {
      r4. do''8 re''8 mi''8
      sol''8 la''8 si''8 re''16 sol''16 mi''8
      re''8 mi''8 fa''8 sol''8 la''8 do''8
      do''4. la''8 sol''8 fa''8
      mi''8 re''8 do''8 la''16 sol''16 fa''8
      sol''8 si''8 la''8 sol''8 fa''8 mi''8
      re''8 re''8 re''8 re''8 re''8 re''8
      sol''8 mi''8 fa''8 sol''8 la''8 do''8
      re''8 do''8 la''8 sol''8 fa''8 mi''8
      sol''8 si''8 la''8 sol''8 fa''8 mi''8
      re''8 re''8 re''8 re''8 re''8 re''8
      r4. do''8 re''8 mi''8
    }
    \bar "|."
  }

  \layout {}

  \midi {}
}
