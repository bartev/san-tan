BARUG 2nd talk
========================================================
JJ Allaire

link to slide deck

http://rpubs.com//jjallaire/rcpp-attributes-bayarea-useR

some videos
Moern c++
not your father's c++
http://goo.gl/N9YbS

Clang: new c++ compiler
http://goo.gl/ixNGP

Learning Rcpp

Rcpp Website: http://www.rcpp.org/

Rcpp Book: http://www.rcpp.org/book/

Rcpp for Beginners: http://goo.gl/If5Oz

Rcpp Attributes: http://goo.gl/PIRN9

http://gallery.rcpp.org articles with full source code



Rccp Attributes
# Seamless C++ Integration with Rcpp Attributes 
JJ Allaire

Attributes are annotations added to C++ source files to provide 
additional information to the compiler . __Rcpp Attributes__ are a new 
feature that provides a high-level syntax for declaring C++ functions 
as callable from R and automatically generating the code required to 
invoke them. The motivation for attributes is several-fold:

* Reduce the learning curve associated with using C++ and R together 
* Eliminate boilerplate conversion and marshaling code wherever possible 
* Seamless use of C++ within interactive R sessions 
* Unified syntax for interactive work and package development

Rcpp supports attributes to indicate that C++ functions should be made 
available as R functions, as well as to optionally specify additional 
build dependencies. The C++ file can then be interactively sourced 
into R using the `sourceCpp` function, which makes all of the exported 
C++ functions immediately available to the interactive R session. As a 
result of eliminating both configuration and syntactic friction, the 
workflow for C++ development in an interactive session now 
approximates that of R code: simply write a function and call it.


This talk will cover the motivation for and implementation of 
attributes as well as review many practical examples of their use.

-----

Why Rccp & Rccp Attributes?
http://www.rccp.org

rccp
write c++ expressions that look like R expressions (similar, not identical)


* R is meant to work with c & fortran.
* There is no copying of data
* All you get is a __pointer__ to the data
