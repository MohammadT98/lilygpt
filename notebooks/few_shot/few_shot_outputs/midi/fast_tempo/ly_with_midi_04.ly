\version "2.24.4"

\language "italiano"

\tempo 4 = 160

\score {
  \new Staff {
    \clef treble
    \key do \major
    \time 4/4

    % Intro
    r4 r8 do''16 re''16 mi''16 fa''16 | sol''8 la''8 si''8 r4 |

    % Theme
    do''4 re''4 mi''4 fa''4 | sol''4 la''4 si''4 do'''4 |

    % Variation
    \tempo 4 = 140
    la''8 sol''16 fa''16 mi''16 re''8 do''8 r4 |

    % Interlude
    r4 r8 si''16 la''16 sol''16 fa''16 | mi''8 re''8 do''8 r4 |

    % Return
    \tempo 4 = 160
    do''4 re''4 mi''4 fa''4 | sol''4 la''4 si''4 do'''4 |

    % Outro
    r4 r4 r2 | \bar "|."
  }

  \layout {}

  \midi {}
}
