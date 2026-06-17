\language "nederlands"
\version "2.24.2"

\header {
  title = "Sinfonia"
  composer = "Vivaldi"
  piece = "Sinfonia"
  opus = ""
  arranger = ""
  copyright = ""
}

global = {
  \key c \major
  \time 4/4
  \tempo "Allegro"
}

violinI = \relative c' {
  \global
  c4 d e f | g a b c' | d4 e f g | a b c' d'
}

violinII = \relative c' {
  \global
  e4 f g a | b c' d e' | f4 g a b | c' d e' f'
}

viola = \relative c' {
  \global
  g4 a b c | d e f g | a4 b c' d' | e f g a
}

oboe = \relative c' {
  \global
  c4 d e f | g a b c' | d4 e f g | a b c' d'
}

bassoon = \relative c' {
  \global
  c2 r | g2 r | d2 r | a2 r
}

\score {
  <<
    \new StaffGroup <<
      \new Staff = "violinI" {
        \clef treble
        \key c \major
        \time 4/4
        \violinI
      }
      \new Staff = "violinII" {
        \clef treble
        \key c \major
        \time 4/4
        \violinII
      }
      \new Staff = "viola" {
        \clef alto
        \key c \major
        \time 4/4
        \viola
      }
      \new Staff = "oboe" {
        \clef treble
        \key c \major
        \time 4/4
        \transpose c c \oboe
      }
      \new Staff = "bassoon" {
        \clef bass
        \key c \major
        \time 4/4
        \transpose c c \bassoon
      }
    >>
  >>
}
