\version "2.24.4"

\language "italiano"

\score {
  <<
    {
      \key sol \major
      \time 4/4
      \voiceOne
      \p
      \fermata sol''8\staccatissimo re''8\staccatissimo mi''8\staccatissimo si''8\staccatissimo
      sol''8\staccatissimo re''8\staccatissimo mi''8\staccatissimo si''8\staccatissimo |
      \slurUp sol''8 re''8 mi''8 si''8 sol''8 re''8 mi''8 si''8 \slurDown |
      \slurUp sol''8. re''16 mi''8. si''16 sol''8. re''16 mi''8. si''16 \slurDown |
      \slurUp re''4~ re''8 mi''16 fa''16 sol''8 re''8 mi''8 si''8 \slurDown |
      sol''2~ sol''4 r4
    }
    \\
    {
      \key sol \major
      \time 4/4
      \voiceTwo
      \mf
      sol''4 r4 sol''4 r4 |
      sol''4~ sol''8 re''8 mi''8 si''8 sol''8 |
      sol''4~ sol''8 re''8 mi''8 si''8 r8 |
      sol''2 r2 |
      sol''4 r4 sol''4 r4
    }
  >>

  \layout {}

  \midi {}
}
