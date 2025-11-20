\version "2.24.4"
\language "italiano"

\score {
  <<
    \new Staff {
      \key fa \major
      \time 4/4
      % Intro
      fad1 | re4 la4 re4 la4 | sol4 la4 sol4 la4 | fad2 | 
      % Verse
      \repeat volta 2 {
        fad4 la4 mi4 re4 | sol4 la4 si4 re4 | fad4 re4 mi4 fa4 | sol4 la4 si4 re4 |
      } 
      % Chorus
      \repeat volta 2 {
        re4 re4 la4 la4 | sol4 sol4 la4 la4 | fad4 re4 mi4 re4 | sol4 la4 si4 re4 |
      }
      % Bridge
      \repeat volta 2 {
        la4 si4 la4 si4 | sol4 la4 sol4 la4 | fad4 re4 mi4 fa4 | sol4 la4 si4 re4 |
      }
      % Outro
      fad2 | re1 | \bar "|."
    }
  >>
}