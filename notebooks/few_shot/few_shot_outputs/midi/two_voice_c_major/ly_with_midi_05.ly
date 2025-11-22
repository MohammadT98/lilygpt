\version "2.24.4"

\language "italiano"

\score {
  <<
    {
      \key do \major
      \time 4/4

      \pp
      \staccato ( do''8 re''8 mi''8 fa''8 )
      sol''8[ sol''8] \staccato
      la''8[ la''8] \staccato

      \mp
      do''8[ re''8 mi''8 fa''8] \accent
      sol''8[ sol''8] \accent
      la''8[ la''8] \accent

      \mf
      do''8[ re''8] \staccato
      sol''8[ sol''8 la''8 fa''8] \staccato
      mi''8[ re''8 mi''8 fa''8] \accent

      \f
      do''8[ re''8] \staccato
      sol''8[ sol''8 la''8 fa''8] \staccato
      mi''8[ re''8 mi''8 fa''8] \accent

      \dim
      do''8~ do''8 \staccato \!

      \cresc
      do''8[ re''8 mi''8 fa''8] \staccato \!

      R4
    }
    \\ {
      \key do \major
      \time 4/4

      \pp
      \staccato ( do8 re8 mi8 fa8 )
      sol8[ sol8] \staccato
      la8[ la8] \staccato

      \mp
      do8[ re8 mi8 fa8] \accent
      sol8[ sol8] \accent
      la8[ la8] \accent

      \mf
      do8[ re8] \staccato
      sol8[ sol8 la8 fa8] \staccato
      mi8[ re8 mi8 fa8] \accent

      \f
      do8[ re8] \staccato
      sol8[ sol8 la8 fa8] \staccato
      mi8[ re8 mi8 fa8] \accent

      \dim
      do8~ do8 \staccato \!

      \cresc
      do8[ re8 mi8 fa8] \staccato \!

      R4
    }
  >>

  \layout {}

  \midi {}
}
