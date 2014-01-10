library(nutshell)
data(field.goals)
names(field.goals)

data.frame(a = c(1,2,3,4,5), b = c(1,2,3,4))


top.bacon.searching.cities <- data.frame(
	city = c("Seattle", "Washington", "Chicago",
			 "New York", "Portland", "St Louis",
			 "Denver", "Boston", "Minneapolis", "Austin",
			 "Philadelphia", "San Francisco", "Atlanta",
			 "Los Angeles", "Richardson"),
	rank = c(100, 96, 94, 93, 93, 92, 90, 90, 89, 87,
			 85, 84, 82, 80, 80)
)

top.bacon.searching.cities
typeof(top.bacon.searching.cities)
class(top.bacon.searching.cities)

a <- 6
top.bacon.searching.cities$rank
top.bacon.searching.cities$city
levels(top.bacon.searching.cities)
sample.formula <- as.formula(y ~ x1 + x2 + x3)
class(sample.formula)
typeof(sample.formula)

ts(data = NA, start = 1, end = numeric(0), frequency = 1,
   deltat = 1, ts.eps = getOption("ts.eps"), class = , names = )

ts(1:22, start = c(2008), frequency = 4)

library(nutshell)
data(turkey.price.ts)
turkey.price.ts

start(turkey.price.ts)
end(turkey.price.ts)
frequency(turkey.price.ts)
delta(turkey.price.ts)

deltat(turkey.price.ts)



dateIStartedWriting <- as.Date("11/24/2011", "%m/%d/%Y")
dateIStartedWriting
today <- Sys.Date()
today - dateIStartedWriting
dateIWasBorn <- as.Date("11/24/1967",  "%m/%d/%Y")
dateIWasBorn


m <- matrix(data=1:12, nrow = 4, ncol = 3, 
		dimnames = list(c("r1", 'r2', 'r3', 'r4'),
						c('c1', 'c2', 'c3')))


attributes(m)
dim(m)
dimnames(m)
colnames(m); rownames(m)
dim(m)<- NULL
m
class(m)
typeof(m)
a <- array(1:12, dim = c(3, 4))
a

b <- 1:12
b
a == b
all.equal(a, b)


identical(a, b)
dim(b) <- c(3, 4)

x <- c(1, 2, 3)
typeof(x)
class(x)

top.bacon.searching.cities
ng.cities
v <- as.integer(c(1, 1, 1, 2, 1, 2, 2, 3, 1))
levels(v) <- c("what", "who", "why")
class(v) <- "factor"
v

class(quote(x))
typeof(quote(x))

x <- 1
y <- 2
z <- 3
v <- quote(c(x, y, z))
eval(v)
v
x <- 5

delayedAssign("w", c(x, y, z))
w

x <- 3
objects()
search()
searchpaths()
parent.env()
environment()
objects()
rm(m)

x <- .GlobalEnv
while (environmentName(x) != environmentName(emptyenv())) {
	print(environmentName(parent.env(x)));
	x <- parent.env(x)
}

env.demo <- function(a, b, c, d) {
	print(objects())
	print(sys.call())
	print(sys.frame())
	print(sys.nframe())
	print(sys.function())
	sys.parent()
	sys.calls()
	sys.frames()
	sys.parents()
	sys.on.exit()
	sys.status()
	parent.frame()
}
env.demo(1, "truck", c(1,2,3,4), pi)




sys.call()
sys.frame()
sys.nframe()
sys.function()



eval(expr, envir = parent.frame(),
		   enclos = if (is.list(envir) || is.pairlist(envir))
		   				parent.frame()
		   			else
		   				baseenv()
)


timethis <- function(...) {
	start.time <- Sys.time();
	eval(..., sys.frame(sys.parent(sys.parent())));
	end.time <- Sys.time();
	print(end.time - start.time)
}

create.vector.of.ones <- function(n) {
	return.vector <- NA;
	timethis <- function(...) {
	start.time <- Sys.time();
	eval(..., sys.frame(sys.parent(sys.parent())));
	end.time <- Sys.time();
	print(end.time - start.time)
}
length(return.vector) <- n
	for (i in 1:n) {
		return.vector[i] <- 1;
	}
	return.vector
}


returned.vector
timethis(returned.vector <- create.vector.of.ones(10000))

length(returned.vector)



timethis <- function(...) {
	start.time <- Sys.time();
	eval.parent(...)
	end.time <- Sys.time();
	print(end.time - start.time)
}

with(data, expr, ...)

example.list <- list(a=1, b=2, c=3)
a + b + c
with(example.list, a + b + c)


within(example.list, d <- a + b + c)


example.list$c

f <- function(x,y) {x + y}
f(1, 2)
g <- function(x, y = 10) {x + y}
g(3)

v <- c(sqrt(1:100))
f <- function(x, ...){print(x); summary(...)}
f("here is the summary for v.", v, digits = 2)
addemup <- function(x, ...) {
	args <- list(...)
	for (a in args) x  <- x + a
	x
}

addemup(1, 1)
addemup(1, 2, 3, 4, 5)


a <- 1:7
sapply(a, sqrt)
sapply(a, cos)
sqrt(1:7)


ApplyToThree <- function(f) {f(3)}
ApplyToThree(function(x) {x * 7})

a <- c(1, 2, 3, 4, 5)


sapply(a, function(x) {x + 1})

(function(x) {x + 1}) (4)
args(sin)
args(`?`)
args(args)
args(lm)
f <- function(x, y = 1, z = 2) {x + y + z}
f.formals <- formals(f)
f.formals$y
f.formals$z


f.formals$y <- 7
formals(f)  <- f.formals
args(f)
f(10)


f <- function(x, y=1, z = 2) {x + y + z}
formals(f) <- alist(x = , y = 100, z = 200)
f
body(f)
f
body(f) <- expression({x * y * z})
rm(list(ls()))
list(ls())
