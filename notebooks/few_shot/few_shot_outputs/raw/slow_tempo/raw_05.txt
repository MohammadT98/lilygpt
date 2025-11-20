\version "2.24.4"
\language "italiano"

\tempo 4 = 60

\score {
  \new Staff {
    \clef treble
    \key do \major
    \time 4/4

    % Sezione A
    do''4 re''4 mi''4 fa''4 \bar "|"
    sol''4 la''4 si''4 do'''4 \bar "|"

    % Sezione B
    re'''8 mi'''8 fa'''8 sol'''8 la'''8 sol'''8 fa'''8 mi'''8 \bar "|"
    re'''4 do'''4 re'''4 do'''4 \bar "|"

    % Sezione C
    R4. \bar "|"

    % Sezione D (con note accidentate)
    dod''8 dod''8 re''8 re''8 mi''8 mi''8 fa''8 fa''8 \bar "|"
    R4. \bar "|"

    % Ripetizione della sezione A
    \repeat volta 2 {
      do''4 re''4 mi''4 fa''4 \bar "|"
      sol''4 la''4 si''4 do'''4 \bar "|"
    }
  }
}