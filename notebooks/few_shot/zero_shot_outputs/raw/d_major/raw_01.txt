\version "2.24.4"
\language "italiano"

\score {
  {
    \key re \major
    \time 4/4

    % Intro
    R1*2
    re'4 mi'4 fa'4 sol'4
    la'4 si'8 re''8 la'8
    re''8 mi''8 fa''8 sol''8 la''8
    re''8 mi''8 fa''8 sol''8

    % Verse
    re'4 re'4 mi'4 mi'4
    fa'4 fa'4 sol'4 sol'4
    la'4 la'4 si'4 si'4
    re''4 re''4 mi''4 mi''4

    % Chorus
    re'8 re'8 mi'8 mi'8 fa'8 fa'8 sol'8 sol'8
    la'8 la'8 si'8 si'8 re''8 re''8 mi''8 mi''8

    \bar "|."
  }
}