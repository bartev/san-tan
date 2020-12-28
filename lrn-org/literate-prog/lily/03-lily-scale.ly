\version "2.20.0"
\header {
  title = "Example"
  composer =  "Emacs!"
}

\score {
  \new Staff \relative c' {
    c8 d e f  g a b c
    d c b a   g f e d
    }
}

\relative c'' {
  g a b c d e f g f e d c b a g a b c d e f g f e d c b a g1
}
