\version "2.24.4"
\language "italiano"

\score {
  \new Staff {
    \key la \minor
    \time 4/4

    % Intro
    R2.*2
    la''8 la''8 la''8 la''8
    re''8 mi''8 fa''8 sol''8
    la''4 si''8 la''8 sol''8 fa''8
    mi''8 re''8 la''8 la''8

    R2.*2

    % Verse
    la''4 la''4 la''4 la''4
    re''8 mi''8 fa''8 sol''8 la''8 si''8
    re''4 re''4 re''4 re''4
    la''8 la''8 la''8 la''8 re''8 mi''8

    R4 r4 r4 r4

    % Chorus
    la''4 la''4 la''4 la''4
    si''8 la''8 sol''8 fa''8 mi''8 re''8
    la''4 la''4 la''4 la''4
    re''8 mi''8 fa''8 sol''8 la''8 si''8

    \bar "|."
  }
}