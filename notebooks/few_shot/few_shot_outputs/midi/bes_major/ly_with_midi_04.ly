\version "2.24.4"

\language "italiano"

\score {
  {
    \key sib \major
    \time 4/4
    \tempo 4 = 120

    % Intro
    r4 r4 r4 r4

    % Verse
    sib'4 la'4 sol'4 fa'4
    sib'2 sib'2
    re'8[ la'8 fa'8 sol'8] re'8[ la'8 fa'8 sol'8]
    re'8[ la'8 fa'8 sol'8] re'8[ la'8 fa'8 sol'8]

    % Interlude
    \repeat volta 2 {
      sol'8[ lad'8 la'8 sol'8] re'8[ la'8 fa'8 sol'8]
    }

    \alternative {
      { re'4 r4 r4 r4 }
      { re'4 r4 r4 r4 }
    }

    % Chorus
    \repeat volta 2 {
      re'2 re'2
    }

    % Bridge in re major
    \key re \major
    re'4 re'4 re'4 re'4
  }

  \layout {}

  \midi {}
}
