\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \time 3/4
    \key do \major
    \tempo "Allegro"

    % Prima frase
    do''8 re''16 mi''16 re''8 do''8
    mi''8 fa''8 sol''8
    do''4 si''8 re''8

    % Seconda frase
    re''4 re''8 do''8
    sol''8 la''8 fa''8
    mi''4 re''8

    % Ripetizione
    \repeat volta 2 {
      do''8 re''8 mi''8 fa''8
      sol''4 re''8
    }

    % Conclusione
    sol''4 sol''8 la''8
    si''4 re''8

    r4 r2
  }

  \layout {}

  \midi {}
}
