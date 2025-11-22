\version "2.24.4"

\score {
  \new Staff {
\language "italiano"

\key la \minor
\time 4/4

<<
{ 
  \f \grace { la''16 } la''4 sol''8 fa''8 mi''8 re''8 la''4
  \p re''8[ sol''16 fa''16 sol''8. mi''16] do''16 sol''16 sol''16
  \mf mi''4. re''8 fa''4. sol''16 sol''16
  \sf \fermata la''8[ la''16 sol''16 fa''8. mi''16] R2
  \> re''8\staccato re''8\staccato re''8\staccato re''8\staccato re''4. \!
  re''8[ re''16 sol''16] re''8[ re''16 sol''16] re''8[ re''16 sol''16] re''8[ re''16 sol''16]
  re''8[ re''16 sol''16] re''8[ re''16 sol''16] re''8[ re''16 sol''16] re''8[ re''16 sol''16]
  la''8[ la''16 sol''16] la''8[ la''16 sol''16] la''8[ la''16 sol''16] la''8[ la''16 sol''16]
}
\\
{
  \f la,2 la,2
  \pp sol,1
  \mf sol,1
  \pp sol,1
  \f la,1
  \pp la,1
  \mf sol,1
  \f la,1
}
>>
}
  \layout {}
  \midi {}
}
