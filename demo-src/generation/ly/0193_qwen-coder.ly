\language "nederlands"
\version "2.24.2"

% Bassoon
bassoon = \relative c' {
  \clef bass
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

% Cello
cello = \relative c {
  \clef bass
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

% Flute
flute = \relative c' {
  \clef treble
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

% Oboe
oboe = \relative c' {
  \clef treble
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

% Recorder
recorder = \relative c' {
  \clef treble
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

% Violin
violin = \relative c' {
  \clef treble
  \key g \minor
  \time 4/4
  r2 g4 b4 c4 d2 |
  e4 f4 g4 a4 b2 |
  c4 d4 e4 f4 g2 |
  a4 b4 c4 d4 e2
}

\score {
  <<
    \new StaffGroup <<
      \new Staff { \bassoon }
      \new Staff { \cello }
      \new Staff { \flute }
      \new Staff { \oboe }
      \new Staff { \recorder }
      \new Staff { \violin }
    >>
  >>
  \layout { }
  \midi { }
}
