# Hastie/Tibshirani p. 122
# Logistic regression models in R
# 2011-11-04

# Followed handout on Logistic regression (with R)
# Christopher Manning
# 4 November 2007


# install.packages('ElemStatLearn')
library(ElemStatLearn)
library(glmnet)

data(SAheart)
head(SAheart)

sahTest <- SAheart[sample(1:nrow(SAheart), 160, replace=FALSE),]
head(sahTest)

# Binomial fit
range(sahTest$chd)
x <- subset(sahTest, select=c(sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age))
y <- subset(sahTest, select=c(chd))
fit=glmnet(x, y, family='binomial')

str(x)

fit1 <- glm(chd ~ sbp + tobacco + ldl + adiposity + famhist + typea + obesity + alcohol + age,
           data=sahTest, family=binomial())
summary(fit1)

fit2 <- glm(chd ~ tobacco + ldl + adiposity + famhist + typea + obesity + alcohol + age,
           data=sahTest, family=binomial())
summary(fit2)

fit3 <- glm(chd ~ tobacco + ldl + adiposity + typea + obesity + alcohol + age,
           data=sahTest, family=binomial())
summary(fit3)

fit4 <- glm(chd ~ tobacco + ldl + typea + obesity + alcohol + age,
           data=sahTest, family=binomial())
summary(fit4)

fit5 <- glm(chd ~ tobacco + ldl + typea + alcohol + age,
           data=sahTest, family=binomial())
summary(fit5)

fit6 <- glm(chd ~ sbp + tobacco + ldl + adiposity + famhist + typea + obesity + alcohol + age,
           data=SAheart, family=binomial())
summary(fit6)

fit7 <- glm(chd ~ sbp + tobacco + ldl + adiposity + famhist + typea + obesity + age,
           data=SAheart, family=binomial())
summary(fit7)

fit <- glm(chd ~ sbp + tobacco + ldl + famhist + typea + obesity + age,
           data=SAheart, family=binomial())
summary(fit)

fit <- glm(chd ~ tobacco + ldl + famhist + typea + obesity + age,
           data=SAheart, family=binomial())
summary(fit)

fit <- glm(chd ~ tobacco + ldl + famhist + typea + age,
           data=SAheart, family=binomial())
summary(fit)

fit <- glm(chd ~ tobacco + ldl + factor(famhist) + age,
           data=SAheart, family=binomial())
summary(fit)
# Test difference in deviances vs Chi2 model for significance
# Chi2 of model - pchisq(residDeviance, DOF-residual)
pchisq(485,44, 457)
# Chi2 of NULL hypothesis
pchisq(596.11, 461)

# Test if w can drop a factor with an anova test (gives an analysis of deviance table)
anova(fit, test='Chisq')

# Test with a 'drop1' test
drop1(fit, test='Chisq')
drop1(fit)
names(fit)
fit$df.residual
fit$boundary
head(fit$data)
fit$weights
head(fit$model)
fit$formula
fit$call
?glm
fit$rank
fit$family
summary(fit)
fit$linear.predictors
head(SAheart)


sah <- transform(SAheart, predChdLogit=predict(fit))
range(sah$predChdLogit)
sah.chd <- subset(sah, chd==1)
head(sah.chd)
sah <- transform(sah, predChdProb=predict(fit, type='response'))
sah <- transform(sah, predChdProb=predict(fit, type='response'), 
                 chdPred=1*(predict(fit) >= 0))
sah <- transform(sah, correct=1*!xor(chd, chdPred))

sig <- function(z) { 1/(1 + exp(-z))}
sah <- transform(sah, sigLgt=sig(predChdLogit))
head(sah)

# How accurate is our model at making predictions?
acc1 <- sum(sah$correct)/nrow(sah)
# could also make a prop.table
t <- table(sah$chd, sah$chdPred)
t.prop <- prop.table(t)
t.prop
accuracy <- sum(diag(t.prop))
accuracy-acc1


sah.fit <- cbind(SAheart, data.frame(fit=fitted(fit)*sah$chd, sah$chd, tot=sah$chd,
                                     diff=sah$chd*(fitted(fit) - 1)))
head(sah.fit)

names(sah)
names(fit)
head(fit$fitted.values)
head(df)

library(ggplot2)
head(sah.fit)
head(SAheart)
head(sah)

# Use ggplot to get BOXPLOTS. Also FACET the plots
p1 <- ggplot(data=sah, aes(x=chd, y=predChdLogit)) + geom_boxplot()
p1

p2 <- ggplot(data=sah, aes(x=chd, y=predChdLogit, group=chd)) + geom_boxplot()
p2

p3 <- ggplot(data=sah, aes(x=chd, y=predChdLogit)) + geom_boxplot() + facet_grid(. ~ chd)
p3

ggplot(data=sah, aes(x=predChdProb, y=predChdLogit)) + facet_grid(. ~ chd) + 
  geom_point() + geom_vline(x=0.5)

ggplot(data=sah, aes(x=predChdProb, y=chdPred)) + geom_point() + facet_grid(. ~ chd)


ggplot(data=sah.fit, aes(x=fit, y=diff)) + geom_point()
head(sah)
table(sah$chd, sah$correct)
t <- table(sah$chd, sah$chdPred)
prop.table(t)


head(SAheart)
sah2 <- cbind(SAheart, predLogit=predict(fit))
sah2 <- cbind(sah2, predFit=fitted(fit))
sah2 <- transform(sah2, SigLogit=sig(predLogit))
# NOTE: fitted(fit) = sigmoid(pred.logit)
head(sah2)
ggplot(data=sah2, aes(x=predLogit, predFit)) + geom_point(alpha = I(1/3)) + facet_grid(. ~ chd)

head(sah)
xtabs(~ sah$chd + sah$chdPred)
xtabs(~ chd + chdPred, data=sah)  # better - used data= , gives better headers

install.packages('Design')
library(Design)