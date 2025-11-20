\version "2.24.4"

\language "italiano"

\score {
  <<
    \new Staff {
      \key re \minor
      \tempo "Allegro" 4=120
      \time 4/4
      \repeat volta 2 {
        r8 re'8 mi'8 fa'8 sol'8 la'8 si'8 re''8 |
        re''8 mi''8 fa''8 sol''8 la''8 si''8 re'''8 r4 |
        r8 re''8 mi''8 fa''8 sol''8 la''8 si''8 re'''8 |
        re'''8 mi'''8 fa'''8 sol'''8 la'''8 si'''8 re''''8 r4 |
        r4 re''8 mi''8 fa''8 sol''4 |
        la''8 si''8 re'''8 mi'''8 fa'''8 sol'''4 |
        r4 re''8 mi''8 fa''8 sol''4 |
        re''8 mi''8 fa''8 sol''8 la''8 si''8 re'''8 |
      } \alternative {
        {
          r4 re''8 mi''8 fa''8 sol''4 |
          la''8 si''8 re'''8 mi'''8 fa'''8 sol'''4 |
        }
        {
          r2 |
        }
      }
    }
  >>

  \layout {}

  \midi {}
}
