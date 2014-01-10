salary <- c(18700000,14626720,14137500,13980000,12916666)
position <- c("QB","QB","DE","QB","QB")
team <- c("Colts","Patriots","Panthers","Bengals","Giants")
name.last <- c("Manning","Brady","Pepper","Palmer","Manning")
name.first <- c("Peyton","Tom","Julius","Carson","Eli")
top.5.salaries <- data.frame(name.last, name.first, team, position, salary)
top.5.salaries
top.5.salaries <- edit(top.5.salaries)

v  <- c("top", "bottom", "front", "back", "left")
v <- edit(v)
v

save(top.5.salaries, file = "top.5.salaries.RData")
save(top.5.salaries, file = "topSalaries")

load("top.5.salaries.RData")

someData <- load("top.5.salaries.RData")
someData

someData <- top.5.salaries
someData

tryData <- load("top.5.salaries.RData")
tryData
ls()
rm("tryData", "someData")
rm("v")

sp500 <- read.csv(paste("http://ichart.finance.yhoo.com/table.csv?",
  "s=%5EGSPC&a=03&b=1&c=1999&d=03&e=1&f=2009&g=m&ignore=.csv"))
  
mySp500 <- read.csv(paste("http://finance.yahoo.com/echarts?s=SPY+Interactive#chart1:symbol=spy;range=my;indicator=volume;charttype=line;crosshair=on;ohlcvalues=0;logscale=on;source=undefined")
)

http://finance.yahoo.com/echarts?s=SPY+Interactive#chart1:symbol=spy;range=my;indicator=volume;charttype=line;crosshair=on;ohlcvalues=0;logscale=on;source=undefined

myUrl <- "http://chart.finance.yahoo.com/z?s=SPY&t=1y&q=&l=&z=l&a=v&p=s&lang=en-US&region=US"
myUrl
mySp500 <- read.csv(paste(myUrl))


system.file("data", "bb.db", package = "nutshell")


x <- c("a", "b", "c", "d", "e")
y <- c("A", "B", "C", "D", "E")
paste(x, y)
paste(x, y, sep = "*", collapse = "--")
top.5.salaries

year <- c(2008, 2008, 2008, 2008, 2008)
rank <- c(1, 2, 3, 4, 5)
more.cols <- data.frame(year, rank)
more.cols
cbind(top.5.salaries, more.cols)


get.quotes <- function(ticker,
				from = (Sys.Date() - 365),
				to = (Sys.Date()),
				interval = "d") {
  # define parts of the URL
  base <- "http://ichart.finance.yahoo.com/table.csv?";
  symbol <- paste("s=", ticker, sep = "");
  # months are numbered from 00 to 11, so format the month correctly
  from.month <- paste("&a=", formatC(
  					as.integer(format(from, "%m")) - 1, 
  						width = 2, 
			     		flag = "0"
			     	),
			     	sep = "");
  from.day <- paste("&b=", format(from, "%d"), sep = "");
  from.year <- paste("&c=", format(from, "%Y"), sep = "");
  to.month <- paste("&d=", formatC(
  					as.integer(format(to, "%m")) - 1,
  						width = 1,
  						flag = "0"
  					),
  					sep = "");
  to.day <- paste("&e=", format(to, "%d"), sep = "");
  to.year <- paste("&f=", format(to, "%Y"), sep = "");
  inter <- paste("&g=", interval, sep = "");
  last <- "&ignore=.csv";
  
  # put together the URL
  url <- paste(base, symbol, from.month, from.day, from.year,
  			to.month, to.day, to.year, inter, last, sep= "");
  
  # get the file
  tmp <- read.csv(url);
  
  #add a new column with ticker symbol labels
  cbind(symbol = ticker, tmp);
  }
  
  
get.multiple.quotes <- function(tkrs,
						from = (Sys.Date() - 365),
  						to = (Sys.Date()),
  						interval = "d") {
  tmp <- NULL;
  for (tkr in tkrs) {
    if (is.null(tmp))
      tmp <- get.quotes(tkr, from, to, interval)
    else
   	  tmp <- rbind(tmp, get.quotes(tkr, from, to, interval))
  }
  tmp
}

dow30.tickers <- c("MMM", "AA", "AXP", "T", "BAC", "BA", "CAT", "CVX", "C",
  "KO", "DD", "XOM", "GE", "GM", "HPQ", "HD", "INTC", "IBM", "JNJ", "JPM",
  "KFT", "MCD", "MRK", "MSFT", "PFE", "PG", "UTX", "VZ", "WMT", "DIS")

Sys.Date()
dow30 <- get.multiple.quotes(dow30.tickers)


dbListFields(con, "Master")

install.packages("nutshell")
library(nutshell)

bbdb <- system.file("data", "bb.db", package = "nutshell")
bbdb

sqlTables(bbdb)
t <- sqlFetch(bbdb, "Teams")


dow30
head(dow30)
summary(dow30)
factors(dow30)
class(dow30$Date)
class(dow30)
dow30$Date <- as.Date(dow30$Date)

dow30$mid <- (dow30$High + dow30$Low) / 2

dow30.transformed <- transform(dow30,
  Date=as.Date(Date),
  mid2 = (High + Low) / 2 )
  
names(dow30.transformed)
levels(dow30.transformed)
factors(dow30)

x <- 1:20
dim(x) <- c(5, 4)
x

apply(X = x, MARGIN = 1, FUN = max)
apply(X = x, MARGIN = 2, FUN = max)


x <- 1:27
dim(x) <- c(3, 3, 3)
x


apply(X = x, MARGIN = 3, FUN = paste, collapse = ",")
apply(X = x, MARGIN = c(1, 2), FUN = paste, collapse = ",")
apply(X = x, MARGIN = 1, FUN = paste, collapse = ",")

x <- as.list(1:5)
lapply(x, function(x) 2^x)

d <- data.frame(x=1:5, y=6:10)
d
lapply(d, function(x) 2^x)
d
lapply(d, FUN=max)
sapply(d, FUN = function(x) 2^x)

library(nutshell)
data(batting.2008)
batting.2008.AB <- transform(batting.2008, AVG = H/AB)
names(batting.2008)
head(batting.2008)
batting.2008.over100AB <- subset(batting.2008.AB, subset = (AB > 100))
count(batting.2008.over100AB)
length(batting.2008.over100AB)
length(batting.2008.AB)
battingavg.2008.bins <- cut(batting.2008.over100AB$AVG, breaks = 10)

battingavg.2008.bins

table(battingavg.2008.bins)

library(lattice)

hat.sizes <- seq(from = 6.25, to = 7.75, by = 0.25)
pants.sizes <- c(30, 31, 32, 33, 34, 36, 38, 40)
shoe.sizes <- seq(from = 7, to = 12)
make.groups(hat.sizes, pants.sizes, shoe.sizes)

names(batting.w.names)
names(batting.2008)
names(batting)
object(nutshell)

batting.w.names.2008 <- batting.w.names[batting.w.names$yearID == 2008]

head(batting.2008)
data(batting.w.names)

batting <- data(Batting)
master <- data(Master)

batting.name.2008 <- batting.2008[c("nameFirst", "nameLast")]
batting.name.2008
names(batting.name.2008)
names(batting.2008)
batting.name.plus.2008 <- cbind(batting.name.2008, batting.2008$HR)

head(batting.name.plus.2008)
names(batting.name.plus.2008)
batting.name.plus.2008 <- cbind(batting.name.2008, 
	batting.2008[c("HR")])

summary(batting.2008)


batting.2008.best <- subset(batting.2008.AB, HR > 5)
batting.2008.best <- subset(batting.2008.best, AB > 100)
batting.2008.best <- subset(batting.2008.best, AVG > 0.35)
batting.2008.best <- subset(batting.2008.best, H > 50)


batting.2008.best

names(batting.2008.best)


nrow(batting.2008.best)
nrow(batting.2008)

sample(1:nrow(batting.2008), 5)
batting.2008[sample(1:nrow(batting.2008), 5), ]

batting.2008$teamID <- as.factor(batting.2008$teamID)
levels(batting.2008$teamID)


sample(levels(batting.2008$teamID), 3)
batting.2008.3teams <- batting.2008[is.element(
	batting.2008$teamID, sample(levels(batting.2008$teamID), 3)),]

summary(batting.2008.3teams$teamID)
summary(batting.2008.3teams)
levels(batting.2008.3teams$teamID)
names(batting.2008.3teams$teamID)

ta <- tapply(X=batting.2008$HR, INDEX = list(batting.2008$teamID),
  FUN = sum)
ta  

tapply(X = (batting.2008$H / batting.2008$AB),
  INDEX = list(batting.2008$lgID),
  FUN = fivenum)

tapply(X = (batting.2008$HR),
  INDEX = list(batting.w.names.2008$lgID,
  				batting.w.names.2008$bats),
  FUN = mean)
  

aggregate(x = batting.2008[ , c("AB", "H", "BB", "2B", "3B", "HR")],
	by = list(batting.2008$teamID), 
	FUN = sum)

rowsum(batting.2008[ , c("AB", "H", "BB", "2B", "3B", "HR")],
	group = batting.2008$teamID)


HR.cnts <- tabulate(batting.2008$HR)
HR.cnts
names(batting.2008)
table(batting.2008$bats)

table(batting.2008[ , c("bats", "throws")])
table(batting.2008[ , c("bats", "throws", "lgID")])


xtabs(~bats+lgID, batting.2008)
xtabs(~bats + throws, batting.2008)

xtabs(~lgID + bats, batting.2008
xtabs(~lgID + bats + throws, batting.2008)

batting.avg.2008 <- transform(batting.2008, AVG = H / AB)
names(batting.2008)
names(batting.avg.2008)

batting.avg.2008.over100AB  <-  subset(batting.avg.2008, 
								subset = (AB > 100))
battingavg.2008.bins <- cut(batting.avg.2008.over100AB$AVG, breaks = 10)

table(battingavg.2008.bins)


m <- matrix(1:10, nrow = 5)

m
t(m)

my.tickers <- c("GE", "GOOG", "AAPL","GS","HPQ")
my.quotes <- get.multiple.quotes(my.tickers,
	from = as.Date("2009-01-01"),
	to = as.Date("2009-03-31"),
	interval = "m")
my.quotes

my.quotes.narrow <- my.quotes[ , c("symbol", "Date", "Close")]
my.quotes.narrow
unstack(my.quotes.narrow, form = Close~symbol)
unstacked <- unstack(my.quotes.narrow, form = Close~symbol)
stack(unstacked)

unstack(my.quotes.narrow, form = symbol~Close)

my.quotes.wide <- reshape(my.quotes.narrow,
	idvar = "Date",
	timevar = "symbol",
	direction = "wide")
my.quotes.wide
attributes(my.quotes.wide)


my.quotes.wide <- reshape(my.quotes.narrow,
	idvar = "symbol",
	timevar = "Date",
	direction = "wide")
my.quotes.wide
my.quotes.narrow

my.quotes.narrow <- my.quotes[ , c("symbol", "Date", "Close")]

unstack(my.quotes.narrow, form = Close~symbol)
unstack(my.quotes.narrow, form = symbol~Close)
unstack(my.quotes.narrow, form = Close~Date)
unstack(my.quotes.narrow, form = Date~symbol)
unstack(my.quotes.narrow, form = symbol~Date)
unstack(my.quotes.narrow, form = Date~Close)

unstacked <- unstack(my.quotes.narrow, form = Close~symbol)
stack(unstacked)
my.quotes.narrow
rm(my.quotes.narro)


my.quotes
my.quotes.narrow
my.quotes.wide <- reshape(my.quotes.narrow, 
	idvar = "Date",
	timevar = "symbol",
	direction = "wide")
my.quotes.wide
my.quotes.wide <- reshape(my.quotes.narrow, 
	idvar = "symbol",
	timevar = "Date",
	direction = "wide")
my.quotes.wide
attributes(my.quotes.wide)
reshape(my.quotes.wide)

my.quotes.oc <- my.quotes[ , c("symbol", "Date", "Close", "Open")]
my.quotes.oc
my.quotes.narrow

reshape(my.quotes.oc,
	idvar = "symbol",
	timevar = "Date",
	direction = "wide")

my.quotes.oc


my.tickers
my.tickers.2 <- c(my.tickers, "GE")
my.tickers.2
my.quotes.2 <- get.multiple.quotes(my.tickers.2, 
  from = as.Date("2009-01-01"),
  to = as.Date("2009-03-31"),
  interval = "m")


duplicated(my.quotes.2)
my.quotes.unique <- my.quotes.2[!duplicated(my.quotes.2), ]
my.quotes.unique
my.quotes.unique  <-  my.quotes.uniqe

w <- c(5, 4, 7, 2, 7, 1)
sort (w)

sort(w, decreasing = TRUE)
length(w)
sort(w)
length(w) <- 7
sort(w, na.last=TRUE)
sort(w, na.last=FALSE)
w
v <- c(11, 12, 13, 15, 14)
order(v)
v[order(v)]
u <- c("pig", "cow", "duck", "horse", "rat")
w <- data.frame(v, u)
w

w[order(w$v),]
w[order(w$u),]
w[order(v),]
w$u
my.quotes[order(my.quotes$Close),]
my.quotes[order(my.quotes$symbol, my.quotes$Date),]
order(my.quotes)
do.call(order, my.quotes)
my.quotes[do.call(order, my.quotes),]

