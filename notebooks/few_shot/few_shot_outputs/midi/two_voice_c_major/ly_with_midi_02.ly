\version "2.24.4"

\score {
  \new Staff {
\language "italiano"
\key do \major
\time 4/4

<<
{ 
  r4
  \p [do''4 re''8 mi''8]
  fa''4 sol''8 mi''8
  la''8 si''8
  do'''8 re'''8
  r4
  do'4 re'8 mi'8
  fa'8 sol'8
  la'8 si'8
  do''8 re''8
  mi''8 fa''8
  sol''8 la''8
  si''8 do'''8
  r4
  \f [do''4 re''8 mi''8]
  fa''8 sol''8
  mi''8 la''8
  si''8 do'''8
  r4
}
\\
{
  r4
  [do'' re'' mi'' fa'']4
  [sol'' la'' si'' do''']4
  r4
  [do' re' mi' fa']4
  [sol' la' si' do'']4
  r4
  do''8 re''8 mi''8 fa''8
  sol''8 la''8 si''8 do'''8
  r4
}
>>
}
  \layout {}
  \midi {}
}
