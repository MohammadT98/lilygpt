\version "2.24.4"
\language "italiano"

\score {
  \new Staff {
    \clef treble
    \key re \minor
    \time 4/4
    \tempo 4 = 100

    % Intro
    R2.
    r4 re8. re16 si8 si8

    % First phrase
    re8 la16 re16 la16 re16 la8 la8
    re8 re8 re8 re8
    re4 r4 r8 re8

    % Second phrase
    re8 dod16 re16 mi16 la8 la8 re8 re8
    re4 r4 r8 re8

    % Repeated section
    \repeat volta 2 {
      re4 re8 re8 re8 re8
      re8 la16 re16 la16 re16 la8 la8
    }

    % Outro
    R2. *2
  }
}