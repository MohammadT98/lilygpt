\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \clef treble
    \key sib \major
    \time 4/4
    \tempo "Allegro" 4=120

    \partial 4
    sol''8 la''8 si''8 re''8 |
    mi''8 fa''8 sol''8 la''8 |
    si''8 sol''8 re''8 mi''8 |
    fa''8 sol''8 la''8 re''8 |

    R4 r8 sib'8 |
    reb'8 mib'8 fab'8 solb'8 |
    lab'8 sib'8 reb'8 mib'8 |
    fab'8 solb'8 lab'8 sib'8 |

    r4 r8 sib''8 |
    reb''8 mib''8 fab''8 solb''8 |
    lab''8 sib''8 reb''8 mib''8 |
    fab''8 solb''8 lab''8 sib''8 |

    \bar "|."
  }

  \layout {}

  \midi {}
}
