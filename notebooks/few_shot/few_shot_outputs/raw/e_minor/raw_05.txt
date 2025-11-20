\version "2.24.4"
\language "italiano"

\score {
  <<
    \new Staff {
      \key mi \minor
      \time 4/4
      \clef treble
      % 8‑measure theme
      mi''8 re''8 mi''8 la''8 |            % 1
      re''8 mi''8 sol''4. sol''8 |          % 2
      fa''8 mi''8 re''8 la''8 |            % 3
      re''8 si''8 la''8 reb''8 |            % 4
      mi''8 re''8 mi''8 la''8 |            % 5
      re''8 mi''8 sol''4. sol''8 |          % 6
      fa''8 mi''8 re''8 la''8 |            % 7
      re''8 si''8 la''8 reb''8 |            % 8
    }
    \new Staff {
      \key mi \minor
      \time 4/4
      \clef bass
      % Simple accompaniment
      mi4 r4 | mi4 r4 | re4 r4 | reb4 r4 |
      mi4 r4 | mi4 r4 | re4 r4 | reb4 r4 |
    }
  >>
}