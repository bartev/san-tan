\version "2.18.2"

                                % set starting point at middle c
\relative c'' {
  a1
  a2 a4 a8 a
  a16 a a a a32 a a a a64 a a a a a a a a2
}

\relative c'' {
  a4 a a4. a8
  a8. a16 a a8. a8 a4.
}

\relative c'' {
  a4 r r2
  r8 a r4 r4. r8
}

\relative c'' {
  \time 3/4
  \tempo "Andante"
  a4 a a
  \time 6/8
  \tempo 4. = 96
  a4. a
  \time 4/4
  \tempo "Presto" 4 = 120
  a4 a a a
}

\relative c' {
  \clef "treble"
  c1
  \clef "alto"
  c1
  \clef "tenor"
  c1
  \clef "bass"
  c1
}

\relative c, {
  \clef "bass"
  \time 3/4
  \tempo "Slowly" 4 = 120
  c2 e8 c'
  \bar "||"
  g'2.
  f4 e d
  \bar "|."
  c4 c, r
}

\relative c'' {
g1 | e1 | c2. c'4 | g4 c | g e | c4 r r2 \bar "|."
}

\relative c'' {
cis4 ees fisis, aeses
}

\relative c'' {
\key d \major
a1 |
\key c \minor
a1 |
}

\relative c'' {
\key d \major
cis4 d e fis
}

\relative c'' {
\key aes \major
aes4 bes b bes
}

\relative c'' {
g4~ g c2~ | c4~ c8 a~ a2 | c4 (c8 a a2)
}

\relative c'' {
d4( c16) cis( d e c cis d) e( d4)
}

\relative c'' {
g4\( g8( a) b( c) b4\)
}

\relative c'' {
c4-^\ff  c-+\mf c-- c-!
c4->\< c-.\p\f  c2-_\pp
}

\relative c'' {
c4-3\< e-5 b-2 a-1\fff
}

\relative c'' {
c2^"espr" a_"legato"
}

\relative c'' {
\partial 4 f8 g8 |
c2 d \bar "|."
}

\relative c'' {
\tuplet 3/2 { f8 g a }
\tuplet 3/4 { c8 r c }
\tuplet 3/2 { f,8 g16[ a g a] }
\tuplet 3/2 { d4 a8 }
}

\relative c'' {
<<
{ a2 g }
{ f2 e }

{ d2 b}
>>
}

\relative c'' {
<<
\new Staff { \clef "treble" c4 }
\new Staff { \clef "bass" c,,4 }
>>
}

