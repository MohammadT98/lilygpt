\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \key mi \minor
    \time 4/4
    % Intro
    r4 mi''8. re''16 mi''8 re''8
    sol''4 sol''8. fad''16 sol''8 fa''8
    re''8 re''8 re''8 re''8
    mi''8 mi''8 mi''8 mi''8
    R1*2
    % Verse
    mi''8 re''16 mi''16 re''8 mi''8
    sol''8 fad''16 sol''16 re''8 fa''8
    re''8 re''8 re''8 re''8
    mi''8 mi''8 mi''8 mi''8
    r2
    % Chorus
    sol''8 sol''8 sol''8 sol''8
    fad''8 fad''8 fad''8 fad''8
    sol''8 sol''8 sol''8 sol''8
    fad''8 fad''8 fad''8 fad''8
  }

  \layout {}

  \midi {}
}
