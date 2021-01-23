### Group Project: EDUCATION IN PORTUGUESE SCHOOLS
#Can we predict students' likelihood of buying math tutor classes based on socio-demographic factors?

getwd()
setwd('C:/Users/stefa/OneDrive/Desktop/Docs/02_HEC/13_Second Semester/02_Data and Decision_Applied Analytics/04_Group project')

stmath = read.csv('student-mat.csv', header = T, sep = ',')


dim(stmath)
names(stmath)

  #0 - Data Manipulation
stmath = as.data.frame(subset(na.omit(stmath), select = -c(G1, G2)))
str(stmath)
attach(stmath)

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

plot(stmath$paid)

################### Tree to find out most relevant variables

names(stmath)

#Creating a sample and training dataset
library(caret)
library(tree)
set.seed(01)
stratid=createDataPartition(stmath$paid, p=0.75, list=F) ## Create the data partition
stmathtree_train=stmath[stratid,] ### Create the training data set
stmathtree_test=stmath[-stratid,] ## Create the testing data set

set.seed(02)
tree.stmath=tree(paid~., 
                 data=stmathtree_train) ## Estimates the classification tree

summary(tree.stmath)
#The misclassification reported is the entropy and it is around 20%
plot(tree.stmath)
text(tree.stmath, pretty=0)

#Cross-validating the tree
cvtree.stmath=cv.tree(tree.stmath, K=10)
summary(cvtree.stmath)

#Finding and implementing optimal tree size
plot(cvtree.stmath$size, cvtree.stmath$dev, type='b')
prune.stmath=prune.tree(tree.stmath, best=5) ## the best parameter limits the size of the tree
plot(prune.stmath) 
text(prune.stmath, pretty=0)
#Most important variables: famsup + reason + famsize + G3 + famsup:G3 + famsup:reason + famsup:reason:famsize

#Test Error
tree.stmath.pred=predict(prune.stmath, stmathtree_test, 
                         type="class") 
library(e1071)
conftabletree = confusionMatrix(as.factor(tree.stmath.pred),as.factor(stmathtree_test$paid), 
                                positive="1")
conftabletree

accuracy_tree = mean(tree.stmath.pred == stmathtree_test$paid)
errorrate_tree = 1-accuracy_tree
sensitivity_tree = unname(conftabletree$table[2,2]/(colSums(conftabletree$table))[2])
specificity_tree =  unname(conftabletree$table[1,1]/(colSums(conftabletree$table))[1])
kappa_tree = unname(conftabletree$overall['Kappa'])

modelfitmetrics_stmathtree = c(accuracy_tree,errorrate_tree,sensitivity_tree,specificity_tree,kappa_tree)
names(modelfitmetrics_stmathtree) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
modelfitmetrics_stmathtree

###Bagging
#install.packages("randomForest", dependencies=TRUE)
library(randomForest)
set.seed(03)
bag.stmath = randomForest(paid~., data=stmathtree_train, 
                          mtry=(ncol(stmathtree_train)-1),importance=T,
                          ntree=500)
bag.stmath
bag.stmath.pred = predict(bag.stmath, stmathtree_test, type = 'class')
conftabletree = confusionMatrix(as.factor(bag.stmath.pred),as.factor(stmathtree_test$paid), 
                                positive="1")

accuracy_bag = mean(bag.stmath.pred == stmathtree_test$paid)
errorrate_bag = 1-accuracy_bag
sensitivity_bag = unname(bag.stmath$confusion[2,2]/(colSums(bag.stmath$confusion))[2])
specificity_bag =  unname(bag.stmath$confusion[1,1]/(colSums(bag.stmath$confusion))[1])
kappa_bag = unname(conftabletree$overall['Kappa'])

modelfitmetrics_stmathbag = c(accuracy_bag,errorrate_bag,sensitivity_bag,specificity_bag,kappa_bag)
names(modelfitmetrics_stmathbag) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
modelfitmetrics_stmathbag

barplot(sort(bag.stmath$importance[,1]), 
        main="Variable importance from Bagging")
sort(bag.stmath$importance[,1])
#Most important variables: famsup + G3 + reason + Medu + Walc + Dalc + Mjob + health + studytime + sex + absences + age + famsize + goout + Pstatus + romantic + schoolsup + higher + school + address + Fedu + traveltime

#RandomForest 
set.seed(04)
rf.stmath = randomForest(paid~., data=stmathtree_train, 
                          mtry=sqrt((ncol(stmathtree_train)-1)),importance=T,
                          ntree=500)
rf.stmath
rf.stmath.pred = predict(bag.stmath, stmathtree_test, type = 'class')
conftabletree = confusionMatrix(as.factor(rf.stmath.pred),as.factor(stmathtree_test$paid), 
                                positive="1")

accuracy_rf = mean(rf.stmath.pred == stmathtree_test$paid)
errorrate_rf = 1-accuracy_rf
sensitivity_rf = unname(rf.stmath$confusion[2,2]/(colSums(rf.stmath$confusion))[2])
specificity_rf =  unname(rf.stmath$confusion[1,1]/(colSums(rf.stmath$confusion))[1])
kappa_rf = unname(conftabletree$overall['Kappa'])

modelfitmetrics_stmathrf = c(accuracy_rf,errorrate_rf,sensitivity_rf,specificity_rf,kappa_rf)
names(modelfitmetrics_stmathrf) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
modelfitmetrics_stmathrf

barplot(sort(rf.stmath$importance[,1]), 
        main="Variable importance from Bagging")
sort(rf.stmath$importance[,1])
#Most important variables: famsup + G3 + reason + Medu + Walc + Dalc + Mjob + health + studytime + sex + absences + age + famsize + goout + Pstatus + romantic + schoolsup + higher + school + address + Fedu + traveltime

#Comparing Tree, Bagging and Random Forest
comp_tree_bag_fr = rbind(modelfitmetrics_stmathtree,modelfitmetrics_stmathbag, modelfitmetrics_stmathrf)
comp_tree_bag_fr


############ For the lines of code below, I tried doing LMs, I am not sure if lines 111 to 142 are necessary to re-clean the data to run lms, 
#I stopped here before trying in depth.

is.numeric(stmath$age)
is.numeric(stmath$Medu)
is.numeric(stmath$Fedu)
is.numeric(stmath$traveltime)
is.numeric(stmath$studytime)
is.numeric(stmath$failures)
is.numeric(stmath$famrel)
is.numeric(stmath$freetime)
is.numeric(stmath$goout)
is.numeric(stmath$Dalc)
is.numeric(stmath$Walc)
is.numeric(stmath$health)
is.numeric(stmath$absences)
is.numeric(stmath$G3)

stmath$schoolsup = as.numeric(stmath$schoolsup)
stmath$sex = as.numeric(stmath$sex)
stmath$guardian = as.numeric(stmath$guardian)
stmath$famsup = as.numeric(stmath$famsup)
stmath$paid = as.numeric(stmath$paid)
stmath$activities = as.numeric(stmath$activities)
stmath$nursery = as.numeric(stmath$nursery)
stmath$higher = as.numeric(stmath$higher)
stmath$internet = as.numeric(stmath$internet)
stmath$romantic = as.numeric(stmath$romantic)
stmath$school = as.numeric(stmath$school)
stmath$address = as.numeric(stmath$address)
stmath$famsize = as.numeric(stmath$famsize)
stmath$Pstatus = as.numeric(stmath$Pstatus)
stmath$Mjob = as.numeric(stmath$Mjob)
stmath$Fjob = as.numeric(stmath$Fjob)
stmath$reason = as.numeric(stmath$reason)

attach(stmath)
stmath$schoolsup = ifelse(schoolsup==1, 0,1)
stmath$sex = ifelse(sex== 1, 0,1)
stmath$guardian = ifelse(guardian == 1, 0,1)
stmath$famsup = ifelse(famsup==1, 0,1)
stmath$paid = ifelse(paid==1, 0,1)
stmath$activities = ifelse(activities==1, 0,1)
stmath$nursery = ifelse(nursery==1, 0,1)
stmath$higher = ifelse(higher==1, 0,1)
stmath$internet = ifelse(internet==1, 0,1)
stmath$romantic = ifelse(romantic==1, 0,1)

#Deriving linear models from tree and bag. Tree should perform better
mod1_lm = lm(paid~famsup+reason+famsize + G3 + famsup:G3 + famsup:reason + famsup:reason:famsize, data = stmath)
summary(mod1_lm)
library(car)
vif(mod1_lm)

mod2_lm = lm(paid~., data = stmath)
summary(mod2_lm)
vif(mod2_lm)

regmetrics=function(model,data, y)
{
  p=predict(model,newdata=data)
  MAPE=sum((abs(y-p))/y)/nrow(data)
  MAD=sum((y-p)^2)/nrow(data)
  MSE=sum(abs(y-p))/nrow(data)
  x = cbind(MAPE,MAD,MSE)
  return (x)
}

set.seed(05)
mod1_folds=createFolds(stmath$paid,k=10)

regmetkfold_mod1_lm = sapply(mod1_folds, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  model=lm(paid~famsup+reason+famsize + G3 + famsup:G3 + famsup:reason + famsup:reason:famsize, data = trainset)
  msp=regmetrics(model, testset, testset$schoolsup)
  names(msp) = c('MAPE', 'MAD', 'MSE')
  return(msp)
})

regmetkfold_mod1_lm
regmetkfold_mod1_lm = apply(regmetkfold_mod1_lm,1,mean)
regmetkfold_mod1_lm

###LDA + QDA

#Preparing the data
attach(stmath)
stmath$schoolsup = as.factor(stmath$schoolsup)
stmath$sex = as.factor(stmath$sex)
stmath$guardian = as.factor(stmath$guardian)
stmath$famsup = as.factor(stmath$famsup)
stmath$paid = as.factor(stmath$paid)
stmath$activities = as.factor(stmath$activities)
stmath$nursery = as.factor(stmath$nursery)
stmath$higher = as.factor(stmath$higher)
stmath$internet = as.factor(stmath$internet)
stmath$romantic = as.factor(stmath$romantic)
stmath$school = as.factor(stmath$school)
stmath$address = as.factor(stmath$address)
stmath$famsize = as.factor(stmath$famsize)
stmath$Pstatus = as.factor(stmath$Pstatus)
stmath$Mjob = as.factor(stmath$Mjob)
stmath$Fjob = as.factor(stmath$Fjob)
stmath$reason = as.factor(stmath$reason)

#LDA
#Holdout Sampling - LDA
library(MASS)

#Creating the training and test data set
library(caret) 
set.seed(06)
stratid=createDataPartition(stmath$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmathlda_train=stmath[stratid,] ## Creating the Training dataset
stmathlda_test=stmath[-stratid,] ## Creating the test data set

mod_lda = lda(paid~., data = stmathlda_train)
mod_lda 

pred_lda_ho = predict(mod_lda, newdata = stmathlda_test)
head(pred_lda_ho$posterior)
head(pred_lda_ho$class)

lda_ho_conftable = table(pred_lda_ho$class, stmathlda_test$paid)
lda_ho_conftable2 = confusionMatrix(as.factor(pred_lda_ho$class),as.factor(stmathlda_test$paid), 
                                positive="1")
lda_ho_conftable2

lda_ho_acc = mean(pred_lda_ho$class==stmathlda_test$paid)
lda_ho_err = 1-lda_ho_acc
lda_ho_sens = unname(lda_ho_conftable2$table[2,2]/(colSums(lda_ho_conftable2$table))[2])
lda_ho_spec =  unname(lda_ho_conftable2$table[1,1]/(colSums(lda_ho_conftable2$table))[1])
lda_ho_kap = unname(lda_ho_conftable2$overall['Kappa'])

lda_ho_modelfitmetrics = c(lda_ho_acc,lda_ho_err,lda_ho_sens,lda_ho_spec,lda_ho_kap)
names(lda_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
lda_ho_modelfitmetrics

#CV - LDA
require(caret)
#10-fold
set.seed(07)
folds_lda_cv10 =createFolds(stmath$paid,k=10)

lda_cv10 = sapply(folds_lda_cv10, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  model=lda(paid~., data = trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

lda_cv10_modelfitmetrics = apply(lda_cv10,1,mean)
lda_cv10_modelfitmetrics

#5-fold
set.seed(08)
folds_lda_cv5 =createFolds(stmath$paid,k=5)

lda_cv5 = sapply(folds_lda_cv5, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  model=lda(paid~., data = trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

lda_cv5_modelfitmetrics = apply(lda_cv5,1,mean)
lda_cv5_modelfitmetrics

#LOOCV
set.seed(09)
folds_lda_cvlo =createFolds(stmath$paid,k=395)

lda_cvlo = sapply(folds_lda_cvlo, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  model=lda(paid~., data=trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

lda_cvlo_modelfitmetrics = apply(lda_cvlo,1,mean)
lda_cvlo_modelfitmetrics

comp_lda = rbind(lda_ho_modelfitmetrics,lda_cv10_modelfitmetrics, lda_cv5_modelfitmetrics, lda_cvlo_modelfitmetrics)
comp_lda

#QDA
#Holdout Sampling - QDA
#Creating the training and test data set
set.seed(10)
stratid=createDataPartition(stmath$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmathqda_train=stmath[stratid,] ## Creating the Training dataset
stmathqda_test=stmath[-stratid,] ## Creating the test data set

mod_qda = qda(paid~famsup+famsize+G3+reason+famsup:G3+famsup:reason,data = stmathqda_train)
mod_qda 

pred_qda_ho = predict(mod_qda, newdata = stmathqda_test)
head(pred_qda_ho$posterior)
head(pred_qda_ho$class)

qda_ho_conftable = table(pred_qda_ho$class, stmathqda_test$paid)
qda_ho_conftable2 = confusionMatrix(as.factor(pred_qda_ho$class),as.factor(stmathqda_test$paid), 
                                    positive="1")
qda_ho_conftable2

qda_ho_acc = mean(pred_qda_ho$class==stmathqda_test$paid)
qda_ho_err = 1-qda_ho_acc
qda_ho_sens = unname(qda_ho_conftable2$table[2,2]/(colSums(qda_ho_conftable2$table))[2])
qda_ho_spec =  unname(qda_ho_conftable2$table[1,1]/(colSums(qda_ho_conftable2$table))[1])
qda_ho_kap = unname(qda_ho_conftable2$overall['Kappa'])

qda_ho_modelfitmetrics = c(qda_ho_acc,qda_ho_err,qda_ho_sens,qda_ho_spec,qda_ho_kap)
names(qda_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
qda_ho_modelfitmetrics

#CV - QDA
#10-fold
set.seed(11)
folds_qda_cv10 =createFolds(stmath$paid,k=10)

qda_cv10 = sapply(folds_qda_cv10, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  model=qda(paid~famsup+famsize+G3+reason+famsup:G3+famsup:reason, data = trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

qda_cv10_modelfitmetrics = apply(qda_cv10,1,mean)
qda_cv10_modelfitmetrics

#5-fold
set.seed(12)
folds_qda_cv5 =createFolds(stmath$paid,k=5)

qda_cv5 = sapply(folds_qda_cv5, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  model=qda(paid~famsup+famsize+G3+reason+famsup:G3+famsup:reason, data = trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

qda_cv5_modelfitmetrics = apply(qda_cv5,1,mean)
qda_cv5_modelfitmetrics

#LOOCV
set.seed(13)
folds_qda_cvlo =createFolds(stmath$paid,k=395)

qda_cvlo = sapply(folds_qda_cvlo, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  model=qda(paid~famsup+famsize+G3+reason+famsup:G3+famsup:reason, data=trainset)
  ppred=predict(model, testset)
  obj=confusionMatrix(as.factor(ppred$class),as.factor(testset$paid), 
                      positive="1")
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

qda_cvlo_modelfitmetrics = apply(qda_cvlo,1,mean)
qda_cvlo_modelfitmetrics

comp_qda = rbind(qda_ho_modelfitmetrics,qda_cv10_modelfitmetrics, qda_cv5_modelfitmetrics, qda_cvlo_modelfitmetrics)
comp_qda

#Comparing LDA and QDA

comp_ldaqda = rbind(comp_lda, comp_qda)
comp_ldaqda



#SVM







