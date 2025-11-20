\version "2.24.4"
\language "italiano"

\score {
  {
    \key mi \minor
    \time 4/4
    R1
    mi8[ re si mi] re8[ fa sol mi]
    sol4 re4
    R2.*2
    re4 mi re mi
    si4 la si la
    \repeat volta 2 {
      mi8[ re si mi] re8[ fa sol mi]
      sol4 re4
    }
    R1
    \bar "|."
  }
}