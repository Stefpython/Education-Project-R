###Setting the Library
setwd("D:/Anis/HEC-Paris/Data and Decision Analysis/Datasets")

###Loading the data
stmath = read.csv('student-mat.csv', header = T, sep = ',')

###Checking everything is okay
dim(stmath)
names(stmath)

###Omitting na, G1, and G2
stmath = as.data.frame(subset(na.omit(stmath)))
stmath = as.data.frame(subset(na.omit(stmath), select = -c(G1, G2)))

###Checking again
str(stmath)
dim(stmath)
attach(stmath)

###Factoring
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

# Creating testing and training data sets
library(MASS)
library(e1071)
library(caret) 
set.seed(14)

###Probit
###HO
stratid=createDataPartition(stmath$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmathprobit_train=stmath[stratid,] ## Creating the Training dataset
stmathprobit_test=stmath[-stratid,] ## Creating the test data set

#Using the probit model on training data
mod_probit_train=glm(paid~.,data=stmathprobit_train,family=binomial(link="probit"))
summary(mod_probit_train)
pred_probit=predict(mod_probit_train, newdata = stmathprobit_test,type = "response")

###Lets look at prediction quality
predclass_probit=rep(0,nrow(stmathprobit_test))
predclass_probit[pred_probit>0.5]=1
probit_conftable=confusionMatrix(as.factor(predclass_probit),as.factor(stmathprobit_test$paid), 
                positive="1")

###Testing
probit_acc = mean(predclass_probit==stmathprobit_test$paid)
probit_err = 1-probit_acc
probit_sens = unname(probit_conftable$table[2,2]/(colSums(probit_conftable$table))[2])
probit_spec =  unname(probit_conftable$table[1,1]/(colSums(probit_conftable$table))[1])
probit_kap = unname(probit_conftable$overall['Kappa'])

probit_ho_modelfitmetrics = c(probit_acc,probit_err,probit_sens,probit_spec,probit_kap)
names(probit_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
probit_ho_modelfitmetrics

############################
###Cross Validation
require(caret)

#10-fold
set.seed(13)
folds_probit_cv10 =createFolds(stmath$paid,k=10)
probit_cv10 = sapply(folds_probit_cv10, function(x)
  {
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_probit_CV10=rep(0,nrow(testset))
  predclass_probit_CV10[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_probit_CV10),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_cv10_modelfitmetrics = apply(probit_cv10,1,mean)
probit_cv10_modelfitmetrics

############################
###5-fold
set.seed(12)
folds_probit_cv5 =createFolds(stmath$paid,k=5)
probit_cv5 = sapply(folds_probit_cv5, function(x)
  
{
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_probit_CV5=rep(0,nrow(testset))
  predclass_probit_CV5[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_probit_CV5),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_cv5_modelfitmetrics = apply(probit_cv5,1,mean)
probit_cv5_modelfitmetrics

############################
###LOOCV
set.seed(11)
folds_probit_cvlo =createFolds(stmath$paid,k=394)
probit_cvlo = sapply(folds_probit_cvlo, function(x)
  
{
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_probit_CVlo=rep(0,nrow(testset))
  predclass_probit_CVlo[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_probit_CVlo),as.factor(testset$paid), 
                      positive="1")
  
metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_cvlo_modelfitmetrics = apply(probit_cvlo,1,mean)
probit_cvlo_modelfitmetrics


###############################
###Combination
comp_probit = rbind(probit_ho_modelfitmetrics,probit_cv10_modelfitmetrics, probit_cv5_modelfitmetrics
, probit_cvlo_modelfitmetrics)
comp_probit

##############################################################################################
###Logit
###HO
set.seed(10)
stratid=createDataPartition(stmath$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmathlogit_train=stmath[stratid,] ## Creating the Training dataset
stmathlogit_test=stmath[-stratid,] ## Creating the test data set

###Using the logit model on training data
mod_logit_train=glm(paid~.,data=stmathlogit_train,family=binomial(link="logit"))
summary(mod_logit_train)
pred_logit=predict(mod_logit_train, newdata = stmathlogit_test,type = "response")

###Lets look at prediction quality
predclass_logit=rep(0,nrow(stmathlogit_test))

predclass_logit[pred_logit>0.5]=1

logit_conftable=confusionMatrix(as.factor(predclass_logit),as.factor(stmathlogit_test$paid), 
                                 positive="1")
###Testing
logit_acc = mean(predclass_logit==stmathlogit_test$paid)
logit_err = 1-logit_acc
logit_sens = unname(logit_conftable$table[2,2]/(colSums(logit_conftable$table))[2])
logit_spec =  unname(logit_conftable$table[1,1]/(colSums(logit_conftable$table))[1])
logit_kap = unname(logit_conftable$overall['Kappa'])

logit_ho_modelfitmetrics = c(logit_acc,logit_err,logit_sens,logit_spec,logit_kap)
names(logit_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')

logit_ho_modelfitmetrics
################################
###Cross Validation
###10-fold
set.seed(9)
folds_logit_cv10 =createFolds(stmath$paid,k=10)
logit_cv10 = sapply(folds_logit_cv10, function(x)
  
{
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_CV10=rep(0,nrow(testset))
  predclass_logit_CV10[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_logit_CV10),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_cv10_modelfitmetrics = apply(logit_cv10,1,mean)
logit_cv10_modelfitmetrics

###########################
###5-fold
set.seed(8)
folds_logit_cv5 =createFolds(stmath$paid,k=5)

logit_cv5 = sapply(folds_logit_cv5, function(x)
  
{
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_CV5=rep(0,nrow(testset))
  predclass_logit_CV5[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_logit_CV5),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_cv5_modelfitmetrics = apply(logit_cv5,1,mean)
logit_cv5_modelfitmetrics

##########################
###LOOCV
set.seed(7)
folds_logit_cvlo =createFolds(stmath$paid,k=394)

logit_cvlo = sapply(folds_logit_cvlo, function(x)
  
{
  trainset=stmath[-x,]
  testset=stmath[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_CVlo=rep(0,nrow(testset))
  predclass_logit_CVlo[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_logit_CVlo),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_cvlo_modelfitmetrics = apply(logit_cvlo,1,mean)
logit_cvlo_modelfitmetrics
###########################################################
###Combination 
comp_logit = rbind(logit_ho_modelfitmetrics,logit_cv10_modelfitmetrics, logit_cv5_modelfitmetrics
                    , logit_cvlo_modelfitmetrics)
comp_logit
################################################################################################
###Comparing Probit and Logit

comp_probit_logit = rbind(comp_probit, comp_logit)
comp_probit_logit
################################################################################################

###Replicating the above with edited dataset, removing the negative coefficient variables
###obtained by Random Forest

###Removing the negative coefficient variables from the stmath dataset; (nursery and traveltime...)
stmath_adj = as.data.frame(subset(na.omit(stmath), select = -c(nursery, freetime, absences, Fedu, 
                                                               goout,school, guardian,romantic,internet, address,traveltime)))
###Checking again
str(stmath_adj)
dim(stmath_adj)
attach(stmath_adj)

set.seed(6)

###Probit_adjusted dataset
###HO
stratid_adj=createDataPartition(stmath_adj$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmath_adj_probit_train=stmath_adj[stratid_adj,] ## Creating the Training dataset
stmath_adj_probit_test=stmath_adj[-stratid_adj,] ## Creating the test data set

#Using the probit model on training data
mod_adj_probit_train=glm(paid~.,data=stmath_adj_probit_train,family=binomial(link="probit"))
summary(mod_adj_probit_train)
pred_adj_probit=predict(mod_adj_probit_train, newdata = stmath_adj_probit_test,type = "response")

###Lets look at prediction quality
predclass_adj_probit=rep(0,nrow(stmath_adj_probit_test))
predclass_adj_probit[pred_adj_probit>0.5]=1
probit_adj_conftable=confusionMatrix(as.factor(predclass_adj_probit),as.factor(stmath_adj_probit_test$paid), 
                                 positive="1")

###Testing
probit_adj_acc = mean(predclass_adj_probit==stmath_adj_probit_test$paid)
probit_adj_err = 1-probit_adj_acc
probit_adj_sens = unname(probit_adj_conftable$table[2,2]/(colSums(probit_adj_conftable$table))[2])
probit_adj_spec =  unname(probit_adj_conftable$table[1,1]/(colSums(probit_adj_conftable$table))[1])
probit_adj_kap = unname(probit_adj_conftable$overall['Kappa'])

probit_adj_ho_modelfitmetrics = c(probit_adj_acc,probit_adj_err,probit_adj_sens,probit_adj_spec,probit_adj_kap)
names(probit_adj_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')
probit_adj_ho_modelfitmetrics
############################
#10-fold
set.seed(5)
folds_probit_adj_cv10 =createFolds(stmath_adj$paid,k=10)
probit_adj_cv10 = sapply(folds_probit_adj_cv10, function(x)
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_adj_probit_CV10=rep(0,nrow(testset))
  predclass_adj_probit_CV10[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_adj_probit_CV10),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_adj_cv10_modelfitmetrics = apply(probit_adj_cv10,1,mean)
probit_adj_cv10_modelfitmetrics
############################
###5-fold
set.seed(4)
folds_probit_adj_cv5 =createFolds(stmath_adj$paid,k=5)
probit_adj_cv5 = sapply(folds_probit_adj_cv5, function(x)
  
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_probit_adj_CV5=rep(0,nrow(testset))
  predclass_probit_adj_CV5[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_probit_adj_CV5),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_adj_cv5_modelfitmetrics = apply(probit_adj_cv5,1,mean)
probit_adj_cv5_modelfitmetrics
############################
###LOOCV
set.seed(3)
folds_probit_adj_cvlo =createFolds(stmath_adj$paid,k=394)
probit_adj_cvlo = sapply(folds_probit_adj_cvlo, function(x)
  
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="probit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_probit_adj_CVlo=rep(0,nrow(testset))
  predclass_probit_adj_CVlo[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_probit_adj_CVlo),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

probit_adj_cvlo_modelfitmetrics = apply(probit_adj_cvlo,1,mean)
probit_adj_cvlo_modelfitmetrics
###############################################################################################
###Computation combination for probit adjusted dataset
comp_probit_adj = rbind(probit_adj_ho_modelfitmetrics,probit_adj_cv10_modelfitmetrics, probit_adj_cv5_modelfitmetrics
                    , probit_adj_cvlo_modelfitmetrics)

comp_probit_adj
###Logit_adjusted dataset
###HO
set.seed(30)
stratid_logit_adj=createDataPartition(stmath_adj$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
stmath_logit_adj_train=stmath_adj[stratid_logit_adj,] ## Creating the Training dataset
stmath_logit_adj_test=stmath_adj[-stratid_logit_adj,] ## Creating the test data set

###Using the logit model on training data
mod_logit_adj_train=glm(paid~.,data=stmath_logit_adj_train,family=binomial(link="logit"))
summary(mod_logit_adj_train)
pred_logit_adj=predict(mod_logit_adj_train, newdata = stmath_logit_adj_test,type = "response")

###Lets look at prediction quality
predclass_logit_adj=rep(0,nrow(stmath_logit_adj_test))

predclass_logit_adj[pred_logit_adj>0.5]=1

logit_adj_conftable=confusionMatrix(as.factor(predclass_logit_adj),as.factor(stmath_logit_adj_test$paid), 
                                positive="1")
###Testing
logit_adj_acc = mean(predclass_logit_adj==stmath_logit_adj_test$paid)
logit_adj_err = 1-logit_adj_acc
logit_adj_sens = unname(logit_adj_conftable$table[2,2]/(colSums(logit_adj_conftable$table))[2])
logit_adj_spec =  unname(logit_adj_conftable$table[1,1]/(colSums(logit_adj_conftable$table))[1])
logit_adj_kap = unname(logit_adj_conftable$overall['Kappa'])

logit_adj_ho_modelfitmetrics = c(logit_adj_acc,logit_adj_err,logit_adj_sens,logit_adj_spec,logit_adj_kap)
names(logit_adj_ho_modelfitmetrics) = c('Accuracy', 'Error rate', 'Sensitivity', 'Specificity', 'Kappa')

logit_adj_ho_modelfitmetrics
################################
###Cross Validation
###10-fold
set.seed(29)
folds_logit_adj_cv10 =createFolds(stmath_adj$paid,k=10)
logit_adj_cv10 = sapply(folds_logit_adj_cv10, function(x)
  
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_adj_CV10=rep(0,nrow(testset))
  predclass_logit_adj_CV10[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_logit_adj_CV10),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_adj_cv10_modelfitmetrics = apply(logit_adj_cv10,1,mean)
logit_adj_cv10_modelfitmetrics

###########################
###5-fold
set.seed(28)
folds_logit_adj_cv5 =createFolds(stmath_adj$paid,k=5)

logit_adj_cv5 = sapply(folds_logit_adj_cv5, function(x)
  
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_adj_CV5=rep(0,nrow(testset))
  predclass_logit_adj_CV5[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor(predclass_logit_adj_CV5),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_adj_cv5_modelfitmetrics = apply(logit_adj_cv5,1,mean)
logit_adj_cv5_modelfitmetrics
##########################
###LOOCV
set.seed(27)
folds_logit_adj_cvlo =createFolds(stmath_adj$paid,k=394)

logit_adj_cvlo = sapply(folds_logit_adj_cvlo, function(x)
  
{
  trainset=stmath_adj[-x,]
  testset=stmath_adj[x,]
  
  
  model=glm(paid~.,data=trainset,
            family=binomial(link="logit"))
  
  ppred=predict(model, newdata = testset,type = "response")
  
  predclass_logit_adj_CVlo=rep(0,nrow(testset))
  predclass_logit_adj_CVlo[ppred>0.5]=1
  
  obj=confusionMatrix(as.factor( predclass_logit_adj_CVlo),as.factor(testset$paid), 
                      positive="1")
  
  metrics=c(obj$overall[1], (1-obj$overall[1]), obj$byClass['Sensitivity'],obj$byClass['Specificity'], unname(obj$overall['Kappa']))
  names(metrics) = c('Accuracy', 'Error rate','Sensitivity','Specificity', 'Kappa')
  return(metrics)
})

logit_adj_cvlo_modelfitmetrics = apply(logit_adj_cvlo,1,mean)
logit_adj_cvlo_modelfitmetrics
###############################################################################################
###Computation Combination for the logit adjusted
comp_logit_adj = rbind(logit_adj_ho_modelfitmetrics,logit_adj_cv10_modelfitmetrics, logit_adj_cv5_modelfitmetrics
                   , logit_adj_cvlo_modelfitmetrics)
comp_logit_adj

###############################################################################################
###Comparing Probit and Logit adjusted dataset

comp_probit_logit_adj = rbind(comp_probit_adj, comp_logit_adj)
comp_probit_logit_adj
###############################################################################################
###Comparing Probit and Logit adjusted and normal dataset
comp_probit_logit_both = rbind(comp_probit_logit, comp_probit_logit_adj)
comp_probit_logit_both
###############################################################################################
###THE END ###

