\language "nederlands"
\version "2.24.0"

\header {
  composer = "Vivaldi"
  title = "Sinfonia"
  subtitle = "for violin, viola, oboe, and bassoon"
}

\layout {
  \context {
    \Score
    \remove "Bar_number_engraver"
  }
}

global = {
  \key c \major
  \time 4/4
  \tempo "Allegro" 4 = 120
}

violin = \new Staff \with {
  instrumentName = "Violin"
  shortInstrumentName = "Vln."
} {
  \global
  \relative c' {
    \set Staff.midiInstrument = "violin"
    c4 d e f | g a b c | d e f g | a b c d |
    e f g a | b c d e | f g a b | c d e f |
    g a b c | d e f g | a b c d | e f g a |
    b c d e | f g a b | c d e f | g a b c |
  }
}

viola = \new Staff \with {
  instrumentName = "Viola"
  shortInstrumentName = "Vla."
} {
  \global
  \relative c' {
    \set Staff.midiInstrument = "viola"
    e4 f g a | b c d e | f g a b | c d e f |
    g a b c | d e f g | a b c d | e f g a |
    b c d e | f g a b | c d e f | g a b c |
    d e f g | a b c d | e f g a | b c d e |
  }
}

oboe = \new Staff \with {
  instrumentName = "Oboe"
  shortInstrumentName = "Ob."
} {
  \global
  \relative c' {
    \set Staff.midiInstrument = "oboe"
    c4 d e f | g a b c | d e f g | a b c d |
    e f g a | b c d e | f g a b | c d e f |
    g a b c | d e f g | a b c d | e f g a |
    b c d e | f g a b | c d e f | g a b c |
  }
}

bassoon = \new Staff \with {
  instrumentName = "Bassoon"
  shortInstrumentName = "Bs."
} {
  \global
  \relative c {
    \set Staff.midiInstrument = "bassoon"
    g4 a b c | d e f g | a b c d | e f g a |
    b c d e | f g a b | c d e f | g a b c |
    d e f g | a b c d | e f g a | b c d e |
    f g a b | c d e f | g a b c | d e f g |
  }
}

\score {
  <<
    \violin
    \viola
    \oboe
    \bassoon
  >>
  \layout { }
  \midi { }
}
