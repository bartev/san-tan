library(nutshell)
data(toxins.and.cancer)
attach(toxins.and.cancer)
plot(total_toxic_chemicals/Surface_Area, deaths_total / Population)
plot(air_on_site / Surface_Area, deaths_lung / Population)
head (toxins.and.cancer)
length(toxins.and.cancer)
toxins.and.cancer
locator(1)
identify(air_on_site/Surface_Area, deaths_lung / Population, State_Abbrev)
plot(air_on_site / Surface_Area, deaths_lung / Population,
  xlab = "Air Release Rae of Toxic Chemicals",
  ylab = "Lung Cancer Death Rate")
text(air_on_site / Surface_Area, deaths_lung / Population,
  labels = State_Abbrev,
  cex = 0.5,
  adj = c(0, -1))

data(batting.2008)
pairs(batting.2008[batting.2008$AB>100, c("H", "R", "SO", "BB", "HR")])


data(turkey.price.ts)
plot(turkey.price.ts)
acf(turkey.price.ts)

data(doctorates)
doctorates
doctorates.m <- as.matrix(doctorates[2:7])
doctorates.m
rownames(doctorates.m) <- doctorates[,1]
doctorates.m

barplot(doctorates.m[1,])
barplot(doctorates.m, 
  beside = TRUE,
  horiz = TRUE,
  legend = TRUE,
  cex.names = 0.75)
  
batting.w.names.2008 <- transform(batting.w.names.2008,
  bat = as.factor(bats),
  throws = as.factor(throws))
cdplot(bats~AVG, data = batting.w.names.2008,
  subset = batting.w.names.2008$AB > 100)

attach(toxins.and.cancer)
plot(total_toxic_chemicals/Surface_Area, deaths_total/Population)
plot(air_on_site/Surface_Area, deaths_lung/Population,
	xlab="Air Release Rate of Toxic Chemicals",
	ylab="Lung Cancer Death Rate")
text(air_on_site/Surface_Area, deaths_lung/Population,	
	labels=State_Abbrev,
	cex=0.5,
	adj=c(0,-1))
locator (1)
identify(air_on_site/Surface_Area, deaths_lung/Population)

head (toxins.and.cancer)
par()
par("bg") <- "red"
par(bg="white")

x <- c(2, 2, 4, 4)
y <- c(2, 4, 4, 2)
polygon(x=c(2, 2, 4, 4), y=c(2, 4, 4, 2))

library(nutshell)
data(batting.2008)
pairs(batting.2008[batting.2008$AB > 100, c("H", "R", "SO", "BB", "HR")])

data(turkey.price.ts)
plot(turkey.price.ts)
acf(turkey.price.ts)

data(doctorates)
doctorates
doctorates.m <- as.matrix(doctorates[2:7])
barplot(doctorates.m[1,])
barplot(doctorates.m, beside=TRUE, horiz=TRUE,legend=TRUE,cex.names=0.75)

barplot(t(doctorates.m), legend=TRUE, ylim=c(0, 66000))

domestic.catch.2006 <- c(7752, 1166, 463, 108)
names(domestic.catch.2006) <- c("Fresh and frozen",
	"Reduced to meal, oil, etc.", "Canned", "Cured")
pie(domestic.catch.2006, init.angle=180, cex=0.6,clockwise = TRUE)
head(domestic.catch.2006)
domestic.catch.2006

batting.w.names.2008 <- transform(batting.w.names.2008,
	bat = as.factor(bats),
	throws = as.factor(throws))
cdplot(bats ~ AVG, data = batting.w.names.2008,
	subset = (batting.w.names.2008$AB > 100))
	
head(batting.2008)

batting.w.names.2008 <- transform(batting.2008,
	bat = as.factor(bats),
	throws = as.factor(throws))
cdplot(bats ~ AVG,
	data = batting.w.names.2008,
	subset = (batting.w.names.2008$AB > 100))
head(batting.w.names.2008)
batting.w.names.2008[AVG <- R/AB]
AVG <- batting.w.names.2008$R / batting.w.names.2008$AB
batting.w.names.2008  <- cbind(batting.w.names.2008, AVG)
head(batting.w.names.2008)
mosaicplot(formula = bats ~ throws, 
	data = batting.w.names.2008,
	color = TRUE)
spineplot(formula = bats ~ throws, 
	data = batting.w.names.2008)

assocplot(table(batting.w.names.2008$bats, 
	batting.w.names.2008$throws),
	xlab = "Throws",
	ylab = "Bats")
	
names(batting.w.names.2008)
class(batting.w.names.2008$throws)
class(batting.w.names.2008$bats)
batting.w.names.2008$bats <- as.factor(batting.w.names.2008$bats)

data(yosemite)
head(yosemite)
dim(yosemite)
yosemite.flipped <- yosemite[,seq(from=253, to= 1)]

yosemite.rightmost <- yosemite[nrow(yosemite) - ncol(yosemite) + 1,]
nrow(yosemite)

#create halfdome subset
halfdome <- yosemite[(nrow(yosemite) - ncol(yosemite) + 1):562,
	seq(from = 253, to = 1)]

persp(halfdome, col = grey(0.25),
	border = NA,
	expand = 0.15,
	theta = 225,
	phi = 20,
	ltheta = 45,
	lphi = 20,
	shade = 0.75)

image(yosemite,
	asp = 253/562,
	ylim = c(1, 0),
	col = sapply((0:32)/32, gray))

contour(yosemite, asp = 253/562, ylim = c(1, 0))my.image

data(batting.2008)
batting.2008 <- transform(batting.2008, PA = AB + BB + HBP + SF + SH)
hist(batting.2008$PA)

hist(batting.2008[batting.2008$PA > 25, "PA"], breaks = 50, cex.main = 0.8)

plot(density(batting.2008[batting.2008$PA > 25, "PA"]))
rug(batting.2008[batting.2008$PA > 25, "PA"])

qqnorm(batting.2008$AB)

batting.2008 <- transform(batting.2008, 
	OBP = (H + BB + HBP) / (AB + BB + HBP + SF))
boxplot(OBP ~ teamID,
	data = batting.2008[batting.2008$PA > 100 & 
	batting.2008$lgID == "AL", ],
	cex.axis = 0.7)

par("bg")

par(mai=c(0.2, 0.4, 0.2, 0.4))
par(mfcol=c(3,2))
pie(c(5,4,3))
plot(x=c(1,2,3,4,5), y=c(1.1, 1.9, 3, 3.9, 6), bty="n", fg="blue", lty=1)
barplot(c(1,2,3,4,5), col="grey95", col.axis="red", col.lab="blue", fg="wheat", bg="green")
barplot(c(1,2,3,4,5), horiz=TRUE)
pie(c(5,4,3,2,1))
plot(c(1,2,3,4,5,6), c(4,3,6,2,1,1))

par(mfcol=c(1,1))

par(mai=c(1,1,0.5, 0.5))
plot(x=c(0, 10), y = c(0, 10))
abline(h=4)
abline(v=3)
abline(a = 1, b = 1)
abline(coef=c(10, -1))
abline(h=0:10, v=0:10)

plot.new()
grid(nx=NULL, ny=NULL, col="blue4", lty="dotted", lwd=par("lwd"), equilogs=TRUE)

polygon(x=c(2,2,4,4), y=c(2,4,4,2))
d <- data.frame(x=c(0:9), y=c(1:10), z=c(rep(c("a", "b"), times=5)))d
d
xyplot(y~x, data = d)
library(lattice)
xyplot(y~x | z, data = d)
xyplot(y~x, data = d, groups=z)

xyplot(y~x|z, data=d,
	panel=function(...){
		panel.abline(a = 1, b = 1)
		panel.xyplot(...)
	})

library(nutshell)
dim(births2006.smpl)

data(births2006.smpl)
head(births2006.smpl)

library(lattice)
births.dow <- table(births2006.smpl$DOB_WK)

dim(births2006.smpl
barchart(births.dow)
head(births.dow)

births2006.dm <- transform(
  births2006.smpl[births2006.smpl$DMETH_REC != "Unknown",],
    DMETH_REC = as.factor(as.character(DMETH_REC)))
dob.dm.tbl <- table(WK=births2006.dm$DOB_WK, MM=births2006.dm$DMETH_REC)

barchart(dob.dm.tbl)

head(dob.dm.tbl)
barchart(dob.dm.tbl, stack = FALSE, auto.key = TRUE)

barchart(dob.dm.tbl, stack = FALSE, auto.key = TRUE,
  horizontal = FALSE,
  groups = FALSE)
  
dob.dm.tbl.alt <- table(WEEK = births2006.dm$DOB_WK,
  MONTH = births2006.dm$DOB_MM,
  METHOD = births2006.dm$DMETH_REC)
dotplot(dob.dm.tbl.alt, stack = FALSE, auto.key = TRUE, groups = TRUE)

data(tires.sus)
head(tires.sus)
names(tires.sus)

dotplot(as.factor(Speed_At_Failure_km_h)~Time_To_Failure | Tire_Type,
  data=tires.sus)
  
histogram(~	DBWT|DPLURAL, data=births2006.smpl)
histogram(~	DBWT|DPLURAL, data=births2006.smpl, layout=c(1, 5))
densityplot(~	DBWT|DPLURAL, data=births2006.smpl, layout=c(1, 5))
densityplot(~	DBWT|DPLURAL, data=births2006.smpl, layout=c(1, 5),
  plot.points = FALSE,
  cex.axis = 0.5)
plot.new()
densityplot(~	DBWT, groups=DPLURAL, 
  data=births2006.smpl, 
  plot.points = FALSE,
  cex.axis = 0.5)
stripplot(~DBWT|DPLURAL, data = births2006.smpl, 
  subset = (DPLURAL=="5 Quintuplet or higher" | 
  			DPLURAL=="4 Quadruplet"),
  jitter.data = TRUE)

qqmath(rnorm(100000))

qqmath(~DBWT|DPLURAL,
  data = births2006.smpl[sample(1:nrow(births2006.smpl), 50000), ],
  pch = 19,
  cex = 0.25,
  subset = (DPLURAL != "5 Quintuplet or higher"))

data(sanfrancisco.home.sales)
head(sanfrancisco.home.sales)
dim(sanfrancisco.home.sales)
qqmath(~price, data=sanfrancisco.home.sales)
qqmath(~log(price), data = sanfrancisco.home.sales)

qqmath(~log(price),
  groups = bedrooms,
  data = subset(sanfrancisco.home.sales,
  				!is.na(bedrooms) & bedrooms > 0 & bedrooms < 7),
  auto.key = TRUE,
  drop.unused.levels = TRUE,
  type = "smooth")
library(Hmisc)
install(Hmisc)
install.packages(Hmisc)

qqmath(~log(price),
  groups = cut2(squarefeet, g = 6),
  data = subset(sanfrancisco.home.sales, !is.na(squarefeet)),
  auto.key = TRUE,
  drop.unused.levels = TRUE,
  type = "smooth")
  
xyplot(price ~ squarefeet, data = sanfrancisco.home.sales)
table(subset(sanfrancisco.home.sales, !is.na(squarefeet), select = zip))

trellis.par.set(fontsize = list(text = 7))
xyplot( price ~ squarefeet | zip,
  data = sanfrancisco.home.sales,
  subset = (zip != 94100 & zip != 94104 & zip != 94108 &
  			zip != 94111 & zip != 94133 & zip != 94158 &
  			price < 4000000 &
  			ifelse(is.na(squarefeet), FALSE, squarefeet < 6000)),
  strip = strip.custom(strip.levels = TRUE))


trellis.par.set(fontsize = list(text = 7))
dollars.per.squarefoot <- mean(
  sanfrancisco.home.sales$price / sanfrancisco.home.sales$squarefeet,
  na.rm = TRUE)

xyplot( price ~ squarefeet | neighborhood,
  data = sanfrancisco.home.sales,
  pch = 19,
  cex = 0.2,
  subset = (zip != 94100 & zip != 94104 & zip != 94108 &
  			zip != 94111 & zip != 94133 & zip != 94158 &
  			price < 4000000 &
  			ifelse(is.na(squarefeet), FALSE, squarefeet < 6000)),
  strip = strip.custom(strip.levels = TRUE,
  	horizontal = TRUE,
  	par.strip.text = list(cex = 0.8)),
  panel = function(...) {
  	panel.abline(a = 0, b = dollars.per.squarefoot);
  	panel.xyplot(...);
  }
)


?cut()

table(cut(sanfrancisco.home.sales$saledate, "month"))
min(sanfrancisco.home.sales$saledate)
max(sanfrancisco.home.sales$saledate)
bwplot(price~cut(saledate, "month"), data = sanfrancisco.home.sales)

bwplot(log10(price)~cut(saledate, "month"), 
  data = sanfrancisco.home.sales,
  scales = list(x = list(rot = 90)))