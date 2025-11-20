\version "2.24.4"
\language "italiano"

\score {
  \new Staff {
    \key do \major
    R1*2
    do''8 re'' mi'' fa'' sol'' la'' si'' do'''
    r2
    sol''8 sol'' sol'' sol'' mi'' sol'' la'' sol''
    r2
    \repeat volta 2 {
      re''8 re'' re'' re'' mi'' re'' do''
      r2
    }
    \alternative {
      { fa''8 fa'' fa'' fa'' mi'' fa'' sol'' fa'' }
      { mi''8 mi'' mi'' mi'' re'' mi'' do'' mi'' }
    }
    r2*3
    do'8 re' mi' fa' sol' la' si' do''
    r2
    sol'8 sol' sol' sol' mi' sol' la' sol'
    r2
    R1*2
    do''8 re'' mi'' fa'' sol'' la'' si'' do'''
    r2
  }
}