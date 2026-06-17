\language "nederlands"
\version "2.24.0"

\header {
  title = "Suite in the Style of Boismortier"
  composer = "Johann-Joseph Boismortier"
  tagline = ##f
}

global = {
  \key d \major
  \time 4/4
  \override Score.BarNumber.break-visibility = ##(#f #f #t)
}

dessus = \relative c' {
  \clef treble
  \global
  \repeat volta 2 {
    a8( bes) c4 d8( e) f4 g8( a) |
    bes4 a8( g) f4 e8( d) c4 |
    bes8( a) g4 f8( e) d4 c8( bes) |
    a4 r8 a' g f e d c4 |
  }
  \repeat volta 2 {
    bes8( a) g4 f8( e) d4 c8( bes) |
    a4 g8( f) e4 d8( c) bes4 |
    a'8( g) f4 e8( d) c4 bes8( a) |
    g4 f8( e) d4 c8( bes) a4 |
  }
}

\score {
  \new StaffGroup <<
    \new Staff \dessus
  >>
  \layout { }
}
