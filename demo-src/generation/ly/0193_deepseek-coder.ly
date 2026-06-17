\language "nederlands"
\version "2.24.0"

\header {
  composer = "de Lalande"
  title = "Sinfonia"
  subtitle = "Basso Part"
}

\paper {
  indent = 0\mm
  left-margin = 15\mm
  right-margin = 15\mm
  top-margin = 10\mm
  bottom-margin = 10\mm
  ragged-last-bottom = ##t
}

global = {
  \key d \major
  \time 4/4
  \tempo 4 = 80
}

upperStrings = \relative c'' {
  \clef treble
  \global
  \set Staff.instrumentName = "Violin "
  \set Staff.shortInstrumentName = "Vl. "
  \repeat volta 2 {
    a4 gis fis e |
    d4. e8 fis4 gis |
    a4 gis fis e |
    d2. r4 |
    a'4 gis fis e |
    d4. e8 fis4 gis |
    a4 gis fis e |
    d2. r4 |
  }
}

lowerStrings = \relative c {
  \clef bass
  \global
  \set Staff.instrumentName = "Bassoon "
  \set Staff.shortInstrumentName = "Bsn. "
  \repeat volta 2 {
    d4 e fis gis |
    a4. b8 cis4 d |
    e4 d cis b |
    a2. r4 |
    d4 e fis gis |
    a4. b8 cis4 d |
    e4 d cis b |
    a2. r4 |
  }
}

\score {
  <<
    \new Staff \upperStrings
    \new Staff \lowerStrings
  >>
  \layout { }
}
