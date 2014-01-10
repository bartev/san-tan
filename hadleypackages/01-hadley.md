Hadley
========================================================

See slides at

http://bit.ly/pkgsrez2


```r
install.packages(c("devtools", "knitr", "Rcpp", "roxygen2", "testthat"))
```

```
## Error: trying to use CRAN without setting a mirror
```


Put all your r code in a directory called 'R'

You write the roxygen comments, and it will convert to Rd file

run
document()
to put documentation in the 'man' directory (Rd file)

check_doc() function runs various checks
dev_help("mypackagename")

vignettes
long-form documentation.
how do I use the functions of this package to do something useful.

add c++ code to the 'src' directory

load_all() will take care of loading r code AND c++ code


Didn't talk about
unit tests ('inst/tests' directory)
data files ('data' directory)

Read other people's packages

install_github('devtools')


## questions
Depending on a package
If you depend on a package, it also loads all if that package's dependencies.
The more functions that get added, the higher the probability of a clash.

using 'import' & namespaces is better.


get daily build of Rstudio, look at new R presentation (based on knitr & r markdown)
