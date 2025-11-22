\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \key do \major
    \time 4/4
    <<
      {
        do''4 \p
        re''8[ sol''16 la''16 fa''8 mi''8] si''4 r16
        re''4 \> la''4[ sol''4] mi''4
        do''8 re''16 mi''16 fa''8 sol''8 la''8 si''4 r16
        re''4~ re''8[ la''8 sol''8] re''8- \> si''4 \fermata
      }
      \\
      {
        do4 \p
        re8[ sol16 la16 fa8 mi8] si8 r16
        re4 \> la4[ sol4] mi4
        do8 re16 mi16 fa8 sol8 la8 si4 r16 r8
        re4~ re8[ la8 sol8] re8- \> si4 \fermata
      }
    >>
  }

  \layout {}

  \midi {}
}
