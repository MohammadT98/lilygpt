\version "2.24.4"

\language "italiano"

\score {
  {
    \clef treble
    \key la \minor
    \time 4/4
    \tempo 4 = 110

    % Tema A
    la'4~ la'8 la'4 ~ la'8
    re'4~ re'8 re'4 ~ re'8
    mi'4~ mi'8 mi'4 ~ mi'8
    do''4~ do''8 do''4 ~ do''8

    % Pausa
    R4 R4

    % Tema B (chords)
    [la' mi' sol']4 [la' mi' sol']4 [la' mi' sol']4 [la' mi' sol']4
    [re' fa' la']4 [re' fa' la']4 [re' fa' la']4 [re' fa' la']4

    % Variazione
    la'8[ re'8 mi'8] do''8[ re''8 mi''8] re''8[ do''8]
    la'8[ re'8 mi'8] do''8[ re''8 mi''8] re''8[ do''8]

    % Coda
    R2.
  }

  \layout {}

  \midi {}
}
