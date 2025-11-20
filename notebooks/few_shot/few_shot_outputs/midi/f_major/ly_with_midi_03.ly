\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \key fa \major
    \time 4/4
    \clef treble
    % Intro
    R1*2
    % Theme
    \repeat volta 2 {
      fa''8 la''16 sol''8. fa''16 sol''8 la''8
      re''8 do''8 mi''8 sol''8
      fa''8 la''8 sol''8 fa''8
      re''8 do''8 mi''8 sol''8
    }
    % Bridge
    sol''8 fa''8 la''8 sol''8
    mi''8 re''8 do''8 sol''8
    fa''8 la''8 sol''8 fa''8
    re''8 do''8 mi''8 sol''8
    % Ending
    R4 r4 r4 r4
  }

  \layout {}

  \midi {}
}
