\version "2.24.4"
\language "italiano"

\tempo 4 = 160

\score {
  \new Staff {
    \clef treble
    \key do \major
    \time 4/4

    % Intro
    R1*2
    do'8 re' mi' fa' | sol' la' si' dod' |
    do'' re'' mi'' fa'' | sol'' la'' si'' dod''

    % Theme
    \repeat volta 2 {
      do''8 re'' mi'' fa'' | sol'' la'' si'' dod'' |
      do'8 re' mi' fa' | sol' la' si' dod' |
      re''8 mi'' fa'' sol'' | la'' si'' dod'' re'' |
      re'8 mi' fa' sol' | la' si' dod' re' |
    }

    % Bridge
    \tempo 4 = 120
    \repeat volta 2 {
      sol''8 la'' si'' dod'' | la'' sol'' re'' mi'' |
      re''8 si'' la'' sol'' | la'' sol'' re'' mi'' |
    }

    % Finale
    do''8 re'' mi'' fa'' | sol'' la'' si'' dod'' |
    do''8 re'' mi'' fa'' | sol'' la'' si'' dod'' |
    R1
  }
}