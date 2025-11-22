\version "2.24.4"
\language "italiano"
\score {
  <<
    { \key sol \major \time 4/4 \mf
      sol''4\accent la''4\accent si''4 do'''4
      | re'''4\accent mi'''4 fa'''4 sol'''4
      | la'''8[ sol'''8] re'''8 mi'''8 sol'''8[ si'''8] la'''8 do'''8
      | do'''4. si''8 la''8 sol''8 sol''4
      | sol''8[ fa''8 la''8] sol''8[ si''8] mi''8[ re''8]
      | re''4 ~ re''8 mi''8 fa''8 sol''8 sol''4
      | la''4 ~ la''8 sol''8 re''8 mi''8 mi''4
      | sol''4 ~ sol''8 mi''8 la''8 sol''8 sol''4
    }
    { \pp
      sol4 la4 si4 do'4
      | re'4 mi'4 fa'4 sol'4
      | la'8[ sol'8] re'8 mi'8 la'8[ sol'8] re'8 mi'8 do'8
      | do'4 ~ do'8 si8 la8 sol8 sol4
      | sol8[ fa8 la8] sol8[ si8] mi8[ re8] do8
      | re4 ~ re8 mi8 fa8 sol8 sol4
      | la4 ~ la8 sol8 re8 mi8 mi4
      | sol4 ~ sol8 mi8 la8 sol8 sol4
    }
  >>
}