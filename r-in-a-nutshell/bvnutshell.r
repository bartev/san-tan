
library(nutshell)
data(field.goals)
names(field.goals)
hist(field.goals$yards)
table(field.goals$play.type)

table(field.goals$play.type)

??regression
example(glm)
help.search("regression")
library(help="grDevices")


vignette("affy")
vignette(all=FALSE)
vignette(all=TRUE)

getOption("defaultPackages")
(.packages())
library()

install.packages(c("tree", "maptree"))
2^1024

x <- 10
y <- c(8, 10, 12, 3, 17)
if (x<y) x else y
if (x>y) x else y

a <- c("a", "a", "a", "a")
b <- c("b", "b", "b", "b")
ifelse(c(TRUE, FALSE, TRUE, FALSE, TRUE), a, b)

i <- 5
repeat {if (i> 25) break else {print(i); i <- i+5}}
i

i <- 5
while (i<= 25) {print(i); i <- i+5}

for (i in seq(from=5, to=25, by=5)) print(i)
library(iterators)

onetofive <- iter(1:5)
nextElem(onetofive)
?foreach


sqrts.1to5 <-foreach(i=1:5) %do% dqrt(i)

o


v <- 100:119
v[5]
v[1:5]
v[c(1,6,11,16)]
v[[3]]


v[-15:-1]
v[-5]
v[-1:-15]
v[-(1:15)]
v[-1:15]

m <- matrix(data=c(101:112), nrow=3, ncol=4)
m

m[3]
m[3,4]
m[1:2, c(2,4)]

m[3:1,]


m[-2:-1]

m[,3:4]

m[,]

a <- array(data=c(101:124), dim=c(2,3,4))
class(a[1,1,])
class(a[1,,])
class(a[1:2, 1:2, 1:2])
class(a[1,1,1,drop=FALSE])

m[1] <- 1000
m[5]

m[1:2, 1:2] <- matrix(c(1001:1004), nrow=2, ncol=2)
m


v <- 1:12
v[15] <- 15
v

rep(c(TRUE,FALSE),10)
v[rep(c(TRUE,FALSE), 10)]

v[c(TRUE, FALSE, FALSE)]
v[as(BOOL,c(1,0,0))]
typeof(TRUE)
?as

as(c(1,0,0),"logical")

rep(c(TRUE,FALSE),10)
v[rep(c(TRUE,FALSE), 10)]
v <- 100:118


v[(v==103)]
v[(v%%3==0)]
v
v[c(TRUE,FALSE,FALSE)]
l[(l>7)]
l <- list(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10)
l

library("nutshell")



l$j
l$f
l[c("a", 'b', 'c')]

dairy <- list(milk='1 gallon', butter="1 pound", eggs=12)
dairy$milk
dairy[['milk']]

dairy[['mil']]
dairy[['mil', exact = FALSE]]



fruit <- list(apples=6, oranges=3, bananas=10)
shopping.list <- list(dairy, fruit)
shopping.list

shopping.list[[c("dairy", "milk")]]
dairy
fruit
shopping.list <- list (dairy, fruit)
shopping.list
shopping.list[[c("dairy", "milk")]]

dairy$milk
shopping.list[[c(1,2)]]I get an error '


v
for (k in seq(along=v)) print(k)
for (k in seq(along=v)) print(v[k])
v

v <- c(.295, .3, 0.2, 0.123)
v

v <- c(v, "zilch")
v
typeof(v)

v <- v[-5]
v
double(v)
typeof(3)
v <- as(v, "double")

v <- c(.295, .3, 0.2, 0.123)
v <- c(v, list(.12, .34, .4))
v <- c(v, list(.12, .34, .4), recursive=TRUE)
v
typeof(v)

1:10
seq(from=5, to=25, by=5)

w <- 1:10
length(w) <- 5
w
length(w) <- 12


l <- list(1,2,3,4,5)
l[1]
l[[1]]

parcel <- list(destination="New York", dimensions=c(2,6,9),
  price=12.95)

parcel$price
parcel$destination
parcel$dimensions


edit(parcel$price)


m <- matrix(data=1:12, nrow=4, ncol=3,
  dimnames=list(c("r1", 'r2', 'r3', 'r4'),
                c('c1', 'c2', 'c3')))
m

m[5]
m[1,2]

a <- array(data=1:24, dim=c(3, 4, 2))
a

eyeColors <- c("brown", "blue", "blue", "green", "brown", "brown", "brown")

eyeColors <- factor(c("brown", "blue", "blue",
                      "green", "brown", "brown", "brown"))
levels(eyeColors)

eyeColors

surveyResults <- factor(
  c("dis", "neut", "str dis", "neut",
    "agr", "neut", "str agr"),
  levels = c("str dis", "dis", "neut", "agr", "str agr"),
  ordered = TRUE)
surveyResults

class(eyeColors)

eyeColors.integerVector <-  unclass(eyeColors)
eyeColors.integerVector
class(eyeColors.integerVector)
class(eyeColors.integerVector) <- "factor"


# 2011-08-25 - Thursday

data.frame(a = c(1,2,3,4,5), b = c(1,2,3,4))

