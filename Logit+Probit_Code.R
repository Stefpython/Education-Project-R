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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

probit_cv5_modelfitmetrics = apply(probit_cv5,1,mean)
probit_cv5_modelfitmetrics

############################
###LOOCV
set.seed(11)
folds_probit_cvlo =createFolds(stmath$paid,k=395)
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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
  return(metrics)
})

logit_cv5_modelfitmetrics = apply(logit_cv5,1,mean)
logit_cv5_modelfitmetrics

##########################
###LOOCV
set.seed(7)
folds_logit_cvlo =createFolds(stmath$paid,k=395)

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
  
  metrics=c(obj$overall[1], obj$byClass['Sensitivity'],obj$byClass['Specificity'] ,(1-obj$overall[1]) , obj$byClass[2])
  names(metrics) = c('Accuracy', 'Sensitivity','Specificity', 'Error rate', 'Kappa')
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

