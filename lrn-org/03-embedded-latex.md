
# Table of Contents

1.  [Here is some LaTex](#orgec84e60)
2.  [Literal exampls](#orgc19dbe2)
    1.  [simple example](#orgd9e8a09)
3.  [From emacs lisp](#orga3dfae8)
4.  [Adding footnote](#org0d56f77)



<a id="orgec84e60"></a>

# Here is some LaTex

The radius of the sun is R<sub>sun</sub> = 6.96 x 10<sup>8</sup> m.

\begin{equation}               % arbitrary environment
x \sqrt{b}                     % even tables, figures
\end{equation}                 % etc

If \(a^2=b\) and \( b = 2 \), then the solution must be
either \[ a = +\sqrt{2} \] or \[ a = -\sqrt{2} \] .


<a id="orgc19dbe2"></a>

# Literal exampls

    Some example from a text file


<a id="orgd9e8a09"></a>

## simple example

    Some example with a colon


<a id="orga3dfae8"></a>

# From emacs lisp

    (defun org-xor (a b)
      "Exclusive or"
      (if a (not b) b))


<a id="org0d56f77"></a>

# Adding footnote

The org homepage <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> looks better than it used to.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> the link is <https://orgmode.org>
