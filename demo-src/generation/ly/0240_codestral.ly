\version "2.22.1"
\language "nederlands"

upper = \relative c'' {
  \time 3/8
  \key c \major
  \tempo 4 = 100
  c4 c c e e d c2
}

lower = \relative c {
  \clef bass
  \time 3/8
  \key c \major
  c2 c c2
}

oboe = \relative c' {
  \clef treble
  \time 3/8
  \key c \major
  r4 c c e e d c2
}

bassoon = \relative c {
  \clef bass
  \time 3/8
  \key c \major
  r2 c c2
}

\score {
  <<
    \new Staff \with {
      instrumentName = "Violin 1"
    } \upper
    \new Staff \with {
      instrumentName = "Viola"
    } \lower
    \new Staff \with {
      instrumentName = "Oboe"
    } \oboe
    \new Staff \with {
      instrumentName = "Bassoon"
    } \bassoon
  >>
  \layout {
    \context {
      \Score
      \override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/8)
      \override SpacingSpanner.uniform-stretching = ##t
    }
  }
}
