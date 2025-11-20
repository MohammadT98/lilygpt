\version "2.24.4"

\language "italiano"

\score {
  \new Staff {
    \key sib \major
    \time 4/4

    sib4 sib4 sib4 sib4
    re4 re4 re4 re4
    reb4 reb4 reb4 reb4
    fa4 fa4 fa4 fa4

    sol4 la4 reb4 re'4
    sib4 re'4 la4 sol4
    re'4 la4 sol4 sib4
    re'4 la4 sol4 re'4

    \repeat volta 2 {
      re4 re4 re4 re4
      re4 re4 re4 re4
    }

    \bar "|."
  }

  \layout {}

  \midi {}
}
