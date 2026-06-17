\version "2.24.1"
\header {
  title = "Sinfonia in Vivaldi Style"
  composer = "Inspired by Vivaldi"
  tagline = ##f
}

\language "nederlands"

global = {
  \key g \major
  \time 4/4
}

violinI = \relative c'' {
  \global
  \clef treble
  \voiceOne
  r8 d e fis g4 a8 g fis e
  d4 r8 d e fis g4 a8 g
  fis4 r8 fis g a b4 c8 b
  a4 r8 a b c d4 e8 d
  c4 r8 c d e fis4 g8 fis
  e4 r8 e fis g a4 b8 a
  g2 r4 r2
}

violinII = \relative c' {
  \global
  \clef treble
  \voiceTwo
  r8 b c d e4 fis8 e d c
  b4 r8 b c d e4 fis8 e
  d4 r8 d e fis g4 a8 g
  fis4 r8 fis g a b4 c8 b
  a4 r8 a b c d4 e8 d
  c4 r8 c d e fis4 g8 fis
  e2 r4 r2
}

viola = \relative c' {
  \global
  \clef alto
  r8 e f g a4 b8 a g f
  e4 r8 e f g a4 b8 a
  g4 r8 g a b c4 d8 c
  b4 r8 b c d e4 f8 e
  d4 r8 d e f g4 a8 g
  f4 r8 f g a b4 c8 b
  a2 r4 r2
}

oboe = \relative c'' {
  \global
  \clef treble
  r8 fis g a b4 c8 b a g
  fis4 r8 fis g a b4 c8 b
  a4 r8 a b c d4 e8 d
  c4 r8 c d e fis4 g8 fis
  e4 r8 e fis g a4 b8 a
  g4 r8 g a b c4 d8 c
  b2 r4 r2
}

bassoon = \relative c {
  \global
  \clef bass
  r8 g a b c4 d8 c b a
  g4 r8 g a b c4 d8 c
  b4 r8 b c d e4 f8 e
  d4 r8 d e f g4 a8 g
  fis4 r8 fis g a b4 c8 b
  a4 r8 a b c d4 e8 d
  c2 r4 r2
}

\score {
  <<
    \new Staff = "violinI" \violinI
    \new Staff = "violinII" \violinII
    \new Staff = "viola" \viola
    \new Staff = "oboe" \oboe
    \new Staff = "bassoon" \bassoon
  >>
  \layout { }
  \midi { }
}
