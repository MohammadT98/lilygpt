\version "2.24.4"
\relative c' {
\key c \major
\time 4/4
\tempo 4 = 100
c4 d4 e4 f4 |
\grace { g8 } c4 d4 e4 f4 |
\acciaccatura { a8 } c4 d4 e4 f4 |
\appoggiatura { b8 } c4 d4 e4 f4 |
\tuplet 3/2 { c8 d e } f4 g4 a4 |
\times 2/3 { c8 d e } f4 g4 a4 |
\ottava 8 { c4 d4 e4 f4 } |
\transpose c c' { c4 d4 e4 f4 } |
}