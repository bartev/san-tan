# Plotting ecdf

library(ggplot2)

# using qplot
qplot(rnorm(1000), stat='ecdf', geom='step')

# using ggplot
df <- data.frame(x = c(rnorm(100, 0, 3), rnorm(100, 0, 10)), g=gl(2, 100), y=rep(1:100,2))
# ecdf
ggplot(df, aes(x, colour=g)) + stat_ecdf()
# in order of df
ggplot(df, aes(x,y, colour=g)) + geom_point()
