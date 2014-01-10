setClass("TimeSeries",
  representation(
    data = "numeric",
    start = "POSIXct",
    end = "POSIXct"
  )
)

my.TimeSeries  <- new("TimeSeries",
  data = c(1, 2, 3, 4, 5, 6),
  start = as.POSIXct("07/01/2009 0:00:00", tz = "GMT",
  					 format = "%m/%d/%Y %H:%M:%S"),
  end = as.POSIXct("07/01/2009 0:05:00", tz = "GMT",
  				   	 format = "%m/%d/%Y %H:%M:%S")
)

my.TimeSeries

setValidity("TimeSeries",
  function(object) {
  	object@start <= object@end &&
  	length(object@start) == 1 &&  	length(object@end) == 1
  }
)
validObject(my.TimeSeries)
good.TimeSeries <- new("TimeSeries",
  data = c(7, 8, 9, 10, 11, 12),
  start = as.POSIXct("07/01/2009 0:06:00", tz = "GMT",
  					format = "%m/%d/%Y %H:%M:%S"),
  end = as.POSIXct("07/01/2009 0:11:00", tz = "GMT",
  					format = "%m/%d/%Y %H:%M:%S")
)

bad.TimeSeries <- new("TimeSeries",
  data = c(7, 8, 9, 10, 11, 12),
  start = as.POSIXct("07/01/2009 0:06:00", tz = "GMT",
  					format = "%m/%d/%Y %H:%M:%S"),
  end = as.POSIXct("07/01/1999 0:11:00", tz = "GMT",
  					format = "%m/%d/%Y %H:%M:%S")
)


period.TimeSeries <- function(object) {
	if (length(object@data) > 1) {
		(object@end - object@start) / (length(object@data) - 1)
	} else {
	  Inf
	}
}

period.TimeSeries(good.TimeSeries)

series <- function(object) {object@data}
setGeneric("series")
series(my.TimeSeries)
series
rm(series)
showMethods("series")


period <- function(object) {object@period}
setGeneric("period")
setMethod(period, signature = c("TimeSeries"), 
  	definition = period.TimeSeries)
showMethods("period")
period(my.TimeSeries)
TimeSeries

setMethod("summary",
  signature = "TimeSeries",
  definition = function(object) {
  	print(paste(object@start,
  				" to ",
  				object@end,
  				sep = "",
  				collapse = ""))
  	print(paste(object@data, sep = "", collapse = ","))
  }
)
summary(my.TimeSeries)
my.TimeSeries

setMethod("[",
  signature = c("TimeSeries"),
  definition = function(x, i, j, ..., drop) {
  	x@data[i]
  }
)
my.TimeSeries[3]


setClass(
	"WeightHistory",
	representation(
	  height = "numeric",
	  name = "character"
	),
	contains = "TimeSeries"
)

john.doe <- new("WeightHistory",
  data = c(170, 169, 171, 168, 170, 169),
  start = as.POSIXct("02/14/2009 0:00:00", tz = "GMT",
    format = "%m/%d/%Y %H:%M:%S"),
  end = as.POSIXct("03/28/2009 0:00:00", tz = "GMT",
     format = "%m/%d/%Y %H:%M:%S"),
  height = 72,
  name = "John Doe"
)

john.doe

setClass("Person",
  representation(
    height = "numeric",
    name = "character"
  )
)

setClass("AltWeightHistory",
  contains = c("TimeSeries", "Person")
)

setClass("Cat",
  representation(
    breed = "character",
    name = "character"
  )
)

setClassUnion("NamedThing",
  c("Person", "Cat")
)

jane.doe <- new("AltWeightHistory",
  data = c(130, 129, 131, 129, 130, 129),
  start = as.POSIXct("02/14/2009 0:00:00", tz = "GMT",
    format = "%m/%d/%Y %H:%M:%S"),
  end = as.POSIXct("03/28/2009 0:00:00", tz = "GMT",
    format = "%m/%d/%Y %H:%M:%S"),
  height = 67,
  name = "Jane Doe"
)

test.jane.doe <- new("Person",
  height = 67,
  name = "Jane Doe"
)
test.jane.doe

is(jane.doe, "NamedThing")
is(jane.doe, "TimeSeries")

is(john.doe, "TimeSeries")
is(john.doe, "NamedThing")

getSlots(john.doe)
extends("AltWeightHistory", "Person")
slotNames(john.doe)
getSlots("Person")

slot("name", john.doe)
slot(name, john.doe)
slot(john.doe, name)
slot("height", jane.doe)

john.doe@name <- "Jack Dope"
john.doe

objects() 
seq(objects())
for (i in seq(objects())) object.size(object(i))

x <- objects()
x

for (i in seq(along=x)) {
	print(x[i], " ", (object.size(x[i]))
}

for (i in seq(along=x)) {
	toPrint <- c(object.size(x[i]), x[i])
	print(toPrint)
}



