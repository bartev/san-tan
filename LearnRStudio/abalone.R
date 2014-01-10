# Read file from website.
# url <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
# download.file(url,destfile='abalone.data')

# Add names
# names(abalone) <- c("Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight", "Rings")

# Write to csv file
# write.csv(abalone, 'abalone.csv', row.names=FALSE)
# 
# Read back from csv
abalone <- read.csv('abalone.csv')

table(abalone$Sex)
plot(Length ~ Sex, data=abalone)

meanLength <- mean(abalone$Length)
model <- lm(Whole.weight ~ Length + Sex, data=abalone)
x <- 1:3
cv <- function(x, na.rm=FALSE){
	sd(x, na.rm=na.rm)/mean(x, na.rm=na.rm)
}

library(ggplot2)
qplot(x=Rings, y=Length, data=abalone)
plot(Length ~ Sex, data=abalone)

library(manipulate)
manipulate(
	plot(Length ~ Rings, data=abalone,
			 axes=axes,
			 cex=cex,
			 pch= if (pch) 19 else 1
			 ),
	axes= checkbox(TRUE, "Show axes"),
	cex= slider(0, 5, initial= 1, step= 0.1, label= "point size"),
	pch= button("Fill points")
	)

# predict(model) returns an error
manipulate({
	if (is.null(manipulatorGetState("model"))){
		fit <- lm(Length ~ Whole.weight, data=abalone)
		manipulatorSetState("model", fit)
		print("hey, I just estimated a model!")
	} else {
		fit <- manipulatorGetState("model")
		print("Now I just retrieved the model from storage")
	}
	plot(abalone$Length, predict(model), col=col)
}
					 , col=picker("red", "green", 'blue')
)

manipulate({
	plot(Length ~ Rings, data=abalone)
	xy <- manipulatorMouseClick()
	if (!is.null(xy)) points(xy$userX, xy$userY, pch=4)
})


# Chapter 3 ---------------------------------------------------------------

plot(Length ~ Whole.weight, data=abalone)
form <- as.formula(paste('Length', 'Whole.weight', sep='~'))

# errata p.52 ------------------------------------------------------------------
# Book says 
# plot(x=form, data=abalone)
# but that gives an error
plot(form, data=abalone)

# using do.call
do.call(plot, list(form, data=abalone))



# data manipulation function ----------------------------------------------

dataplot <- function(dat) {
	name <- sys.call()[[2]]
	vars <- as.list(names(dat))
	e <- new.env()
	e$data <- name
	manipulate(
		{
			form <- as.formula(paste(y,x,sep='~'))
			plot(form, data=dat, main=as.character(name), las=1)
			e$form <- form
		},
		x=do.call(picker, c(vars, initial=vars[1])),
		y=do.call(picker, c(vars, initial=vars[2]))
		)
	invisible(e)
}

# execute the function
f <- dataplot(abalone)

do.call(plot, as.list(f))
plot(x=f$form, data=f$data)