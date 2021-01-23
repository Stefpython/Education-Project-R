stmath=student.mat
stmath = as.data.frame(subset(na.omit(stmath), select = -c(G1, G2)))
str(stmath)
attach(stmath)
dim(stmath)


##factor
stmath$schoolsup = as.factor(ifelse(schoolsup=='no', 0,1))
stmath$sex = as.factor(ifelse(sex== 'M', 0,1))
stmath$guardian = as.factor(ifelse(guardian == 'father', 0,1))
stmath$famsup = as.factor(ifelse(famsup=='no', 0,1))
stmath$paid = as.factor(ifelse(paid=='no', 0,1))
stmath$activities = as.factor(ifelse(activities=='no', 0,1))
stmath$nursery = as.factor(ifelse(nursery=='no', 0,1))
stmath$higher = as.factor(ifelse(higher=='no', 0,1))
stmath$internet = as.factor(ifelse(internet=='no', 0,1))
stmath$romantic = as.factor(ifelse(romantic=='no', 0,1))

stmath$school = as.factor(stmath$school)
stmath$address = as.factor(stmath$address)
stmath$famsize = as.factor(stmath$famsize)
stmath$Pstatus = as.factor(stmath$Pstatus)
stmath$Mjob = as.factor(stmath$Mjob)
stmath$Fjob = as.factor(stmath$Fjob)
stmath$reason = as.factor(stmath$reason)

####



stmath$paid = as.factor(stmath$paid)
class(stmath$paid)


set.seed(11) ## Ensure that results are reproducible

library(caret) 
stratid=createDataPartition(stmath$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmath_strat_train=stmath[stratid,] ## Creating the Training dataset
stmath_strat_test=stmath[-stratid,] ## Creating the test data set

### Fitting a simple regression tree

library(tree)
summary(stmath_strat_train$paid)
set.seed(66)
tree.stmath=tree(paid~., data=stmath_strat_train) ## Fit the tree on the training dataset
summary(tree.stmath) 
## Note that only few variables have been used in constructing
## the dataset

## Lets plot the tree
plot(tree.stmath)
text(tree.stmath, pretty=0)

## Evaluate the test error in this case using tree
tree.pred=predict(tree.stmath,stmath_strat_test,type="class")
summary(tree.pred)
summary(stmath_strat_test$paid)
summary(stmath_strat_train$paid)
prop.table(table(tree.pred, stmath_strat_test$paid))
mean(tree.pred==stmath_strat_test$paid) 
library(e1071)
conftabletree1 = confusionMatrix(as.factor(tree.pred),as.factor(stmath_strat_test$paid), 
                                positive="1")
conftabletree1
## Cross validating the tree
cvtree.stmath=cv.tree(tree.stmath, K=10)

## the important parameters to consider is how increasing the size of the tree
## reduces the deviance in the fit

plot(cvtree.stmath$size, cvtree.stmath$dev, type='b')
prune.stmath=prune.tree(tree.stmath, best=5) ## the best parameter limits the size of the tree
plot(prune.stmath) 
text(prune.stmath, pretty=0)

tree.stmath.pred=predict(prune.stmath, stmath_strat_test, 
                         type="class")
summary(tree.stmath.pred)
library(e1071)
conftabletree = confusionMatrix(as.factor(tree.stmath.pred),as.factor(stmath_strat_test$paid), 
                                positive="1")
conftabletree



## Next lets prune the tree and cross validate using cost complexity to see if we can
## improve fit

set.seed(11)

cv.stmath.tree=cv.tree(tree.stmath) ## cv tree

## size is the size of the tree, dev is the deviance correction, k is the complexity parameter
## The default pruning occurs using deviance as the criteria
## we can change that to misclassification rate by using an additional parameter called FUN

cv.stmath.tree.mis=cv.tree(tree.stmath, FUN=prune.misclass)

## LEts now plot the error rate as a function of size and cost

par(mfrow=c(1,2))

plot(cv.stmath.tree.mis$size, cv.stmath.tree.mis$dev, type="b")
plot(cv.stmath.tree.mis$k, cv.stmath.tree.mis$dev, type="b")

## Cross validation results suggest that maybe the right size is around 7-8 nodes
## lets prune the tree to 6 nodes and see if we can improve the results

prune.stmath2=prune.misclass(tree.stmath, best=2)
plot(prune.stmath2) 
text(prune.stmath2, pretty=0)
## Lets check our out of sample predictions

prune.pred2=predict(prune.stmath2, stmath_strat_test, type="class")
table(prune.pred2, stmath_strat_test$paid)
mean(prune.pred2==stmath_strat_test$paid)


conftabletree2 = confusionMatrix(as.factor(prune.pred2),as.factor(stmath_strat_test$paid), 
                                positive="1")
conftabletree2
## not really much of an improvement with the tree

### We could next try bagging

library(randomForest)

set.seed(11)
bag.stmath=randomForest(stmath_strat_train$paid~.,
                    data=stmath_strat_train, 
                    mtry=(ncol(stmath_strat_train)-1),
                    importance=T)
bag.stmath

## Lets get the prediction error now
set.seed(66)
pred.bag.stmath=predict(bag.stmath, stmath_strat_test, type="class")
mean(pred.bag.stmath==stmath_strat_test$paid)


conftablebag = confusionMatrix(as.factor(pred.bag.stmath),as.factor(stmath_strat_test$paid), 
                                 positive="1")
conftablebag

par(mfrow=c(1,1))
barplot(sort(bag.stmath$importance[,1]))
sort(bag.stmath$importance[,1])

### Try the same for the random forest
set.seed(11)
bag.stmath.sqrt=randomForest(stmath_strat_train$paid~.,
                    data=stmath_strat_train, 
                    mtry=sqrt(ncol(pl_strat_train)-1), 
                    importance=T,
                    ntrees=5000)
bag.stmath.sqrt

## Lets get the prediction error now
pred.bag.stmath.sqrt=predict(bag.stmath.sqrt, stmath_strat_test, type="class")
mean(pred.bag.stmath.sqrt==stmath_strat_test$paid)

conftablebag.sqrt = confusionMatrix(as.factor(pred.bag.stmath.sqrt),as.factor(stmath_strat_test$paid), 
                                positive="1")
conftablebag.sqrt

par(mfrow=c(1,1))
barplot(sort(bag.stmath.sqrt$importance[,1]))
sort(bag.stmath.sqrt$importance[,1])



## The End
######################
##############
##########
######

stmath_numeric= student.mat
stmath_numeric= as.data.frame(subset(na.omit(stmath_numeric), select = -c(G1, G2)))
stmath_numeric$schoolsup = ifelse(schoolsup=='no', 0,1)
stmath_numeric$sex = ifelse(sex== 'M', 0,1)
stmath_numeric$guardian = ifelse(guardian == 'father', 0,1)
stmath_numeric$famsup = ifelse(famsup=='no', 0,1)
stmath_numeric$paid = ifelse(paid=='no', 0,1)
stmath_numeric$activities = ifelse(activities=='no', 0,1)
stmath_numeric$nursery = ifelse(nursery=='no', 0,1)
stmath_numeric$higher = ifelse(higher=='no', 0,1)
stmath_numeric$internet = ifelse(internet=='no', 0,1)
stmath_numeric$romantic = ifelse(romantic=='no', 0,1)

mod_lm=lm(paid~.,data=stmath_numeric)
summary(mod_lm)

step(lm(paid~.,data=stmath_numeric))

mod_lm_step =lm(formula = paid ~ sex + reason + guardian + studytime + failures + 
                  famsup + nursery + higher + internet + Walc, data = stmath_numeric)
summary(mod_lm_step)
p_lm=predict(mod_lm_step)
summary(p_lm)
plot(lm(paid ~ sex + reason + guardian + studytime + failures + 
          famsup + nursery + higher + internet + Walc, data = stmath_numeric))
library(tseries)
jarque.bera.test(mod0$residuals)
plot(mod_lm_step$residuals)

library(lmtest)
bptest(mod_lm_step)
