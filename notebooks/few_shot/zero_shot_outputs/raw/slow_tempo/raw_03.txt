\version "2.24.4"
\language "italiano"

\score {
  \tempo 4 = 60
  \time 4/4
  {
    % A section
    r4 do''8 mi''16 sol''16
    sol''8 la''8 sol''8 mi''8
    re''4 re''8 re''8
    [do'' re'' mi''] sol''4
    re''8 re''16 mi''16 fa''16 sol''16
    sol''8 la''16 sol''16 fa''16 mi''16
    re''8 re''8 re''8 re''8
    sol''4 r4

    % B section
    do''4 re''4 mi''4 fa''4
    sol''8 la''16 sol''16 fa''16 mi''16 re''8
    re''4 re''4
    sol''8 si''16 sol''16 la''16 sol''16 fa''8

    % Repeat A
    r4 do''8 mi''16 sol''16
    sol''8 la''8 sol''8 mi''8
    re''4 re''8 re''8
    [do'' re'' mi''] sol''4
    re''8 re''16 mi''16 fa''16 sol''16
    sol''8 la''16 sol''16 fa''16 mi''16
    re''8 re''8 re''8 re''8
    sol''4 r4

    \bar "|."
  }
}