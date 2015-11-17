% See https://fonts.openlilylib.org/lilyjazz/
% https://fonts.openlilylib.org/docs.html#stylesheets

% To use the font globally, put the following (minimal) code 
% at the top of your .ly file

% NOTE: The #:factor (/ staff-height pt 20) argument is only
% needed if you are working at staff-height other than 20pt.

#(set-global-staff size 16)  % this MUST go PRIOR to defining the fonts!!!
\include "./stylesheets/lilyjazz.ily"
\include "./stylesheets/jazzchords.ily"
% \paper {
%   #(define fonts
%     (set-global-fonts
%      #:music "lilyjazz"
%      #:brace "lilyjazz"
%      #:roman "lilyjazz-text"
%      #:sans "lilyjazz-chord"
%      #:typewriter "lilyjazz-chord"
%      #:factor(/ staff-height pt 20)
%    ))
% }

% \layout {
%   \override Staff.Tie.thickness = #3
%   \override Staff.Slur.thickness = #3
%   \override Staff.PhrasingSlur.thickness = #3
%   \override Score.Hairpin.thickness = #2
%   \override Score.Stem.thickness = #2
%   \override Score.TupletBracket.thickness = #2
%   \override Score.VoltaBracket.thickness = #2
%   \override Staff.BarLine.hair-thickness = #2
%   \override Staff.BarLine.thick-thickness = #4
%   \override Staff.MultiMeasureRest.hair-thickness = #3
%   \override Staff.MultiMeasureRestNumber.font-size = #2
%   \override LyricHyphen.thickness = #3
%   \override LyricExtender.thickness = #3
%   \override PianoPedalBracket.thickness = #2
% }

% showStartRepeatBar = { 
%   \once \override Score.BreakAlignment.break-align-orders =
%         #(make-vector 3 '(instrument-name
%                           left-edge
%                           ambitus
%                           breathing-sign
%                           clef
%                           key-signature
%                           time-signature
%                           staff-bar
%                           custos))
%       \once \override Staff.TimeSignature.space-alist =
%         #'((first-note . (fixed-space . 2.0))
%            (right-edge . (extra-space . 0.5))
%            ;; free up some space between time signature
%            ;; and repeat bar line
%            (staff-bar . (extra-space . 1)))
% }

% %=> http://lsr.di.unimi.it/LSR/Item?id=753
% #(define (white-under grob) (grob-interpret-markup grob 
%   (markup #:vcenter #:whiteout #:pad-x 1 (ly:grob-property grob 'text))))

% inlineMMR = {
%   \once \override MultiMeasureRest.layer = #-2
%   \once \override MultiMeasureRestNumber.layer = #-1
%   \once \override MultiMeasureRestNumber.Y-offset = #0
%   \once \override MultiMeasureRestNumber.stencil = #white-under
%   \once \override MultiMeasureRest.rotation = #'(2 0 0)
% }


\header{
  title = "A scale in LilyPond"
  composer = "Bartev Vartanian"
  tagline = "Some tagline"
}

\relative {
  c'1 d e f g a b c
}

\score {
  \new Staff {
    % \jazzOn
    c'4 c' \tuplet 3/2 { d'8-- es'-- e'-- } g'4 ~ |
    g'4 r r8 f'-^ \noBeam es' c'-> \bar "|."
  }
}
