setwd("~/Rmac/MASS")

x <- rt(250, df=9)
par(pty='s')
qqnorm(x)
qqline(x)
library(ggplot2)
qplot(x)

library(MASS)
x <- rgamma(100, shape=5, rate=0.1)
fitdistr(x, 'gamma')

x2 <- rt(250, df=9)
fitdistr(x2, 't', df=9)
fitdistr(x2, 't')
fitdistr(x2, 't', df=20)

contam <- rnorm(100, 0, (1 + 2*rbinom(100, 1, 0.05)))
qplot(contam)
nm <- rnorm(100, 0)

mybinom <- rbinom(100, 1, 0.05)
qplot(mybinom)
plot(mybinom)
plot(contam)

x <- 1:12
sample(x)

sample(x, replace=TRUE)
sample(1:5, 100, replace=T)

sort(sample(1:6, 2, replace=T))
diceRolls <- 

