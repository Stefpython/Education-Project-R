getwd()
setwd('~/Desktop')

stmath = read.csv('student-mat.csv', header = T, sep = ',')


dim(stmath)
names(stmath)

#0 - Data Manipulation
stmath = as.data.frame(subset(na.omit(stmath)))
stmath = stmath[, -c(9, 10, 11, 31, 32)]
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

stmath$school = as.factor(ifelse(school=='GP', 0,1))
stmath$address = as.factor(ifelse(address=='R', 0,1))
stmath$famsize = as.factor(ifelse(famsize=='LE3', 0,1))
stmath$Pstatus = as.factor(ifelse(Pstatus=='A', 0,1))

## kNN

## Normalizing our data
normalize = function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

non_num_var = stmath[, c(1,2,4,5,6,9,13,14,15,16,17,18,19,20)]
num_var = stmath[, c(3,7,8,10,11,12,21,22,23,24,25,26,27,28)]

Pnorm = as.data.frame(lapply(num_var, normalize))
Pnormed=cbind(Pnorm, non_num_var)
names(Pnormed)

# Creating testing and training data sets

library(caret) 
set.seed(14)
stratid=createDataPartition(Pnormed$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
Normstmath_train=Pnormed[stratid,] ## Creating the Training dataset
Normstmath_test=Pnormed[-stratid,] ## Creating the test data set

## Run the KNN algorithm on the Training data set

library(class)

knn_mod = knn(train=Normstmath_train[,-23], test=Normstmath_test[,-23], cl=Normstmath_train[,23], k=16)

## Test model predictions using the confusion matrix

library(e1071)
confusionMatrix(as.factor(knn_mod),as.factor(Normstmath_test[,23]), 
                positive="1")

## KNN with cross validation
folds=createFolds(Pnormed$paid,k=10)

knn_cvaccuracy = sapply(folds, function(x){
  trainset=Pnormed[-x,]
  testset=Pnormed[x,]
  model=knn(trainset[,-23], test=testset[,-23], cl=trainset[,23], k=16)
  obj=confusionMatrix(as.factor(model),as.factor(testset[,23]), 
                      positive="1")
  metrics=c(obj$overall[1], obj$overall[2], obj$byClass[1], obj$byClass[2])
  return(metrics)
})

knn_cvaccuracy

### Naive Bayes Classifier

library(e1071)

## Holdout sampling apporach

Normstmath_train_nb=Normstmath_train[,-23] ## Pulling the features out from the train set
Normstmath_test_nb=Normstmath_test[,-23] ## Pulling the features set out of the test set

nbmath=naiveBayes(Normstmath_train_nb, as.factor(Normstmath_train[,23])) ## Run the Naive Bayes model

nbpred=predict(nbmath, Normstmath_test_nb, type="class") ## Generate Test Predictions


## Generate the confusion matrix

confusionMatrix(as.factor(nbpred),as.factor(Normstmath_test[,23]), 
                positive="1")

## Naive Bayes with cross validation
nb_folds=createFolds(Pnormed$paid,k=10)

nb_cvaccuracy = sapply(folds, function(x){
  trainset=Pnormed[-x,]
  testset=Pnormed[x,]
  
  trainx=trainset[,-23]
  testx=testset[,-23]
  
  model=naiveBayes(trainx, as.factor(trainset[,23]))
  npred=predict(model, testx, type="class")
  
  obj=confusionMatrix(as.factor(npred),as.factor(testset[,23]), 
                      positive="1")
  metrics=c(obj$overall[1], obj$overall[2], obj$byClass[1], obj$byClass[2])
  return(metrics)
})

nb_cvaccuracy

## Running kNN and Naive Bayes with variables removed

remove(stmath)

stmath = read.csv('student-mat.csv', header = T, sep = ',')


dim(stmath)
names(stmath)

#0 - Data Manipulation
stmath = as.data.frame(subset(na.omit(stmath)))
stmath = stmath[, -c(1, 8, 9, 10, 11, 12, 13, 20, 22, 23, 25, 26, 30, 31, 32)]
str(stmath)
attach(stmath)

stmath$schoolsup = as.factor(ifelse(schoolsup=='no', 0,1))
stmath$sex = as.factor(ifelse(sex== 'M', 0,1))
stmath$famsup = as.factor(ifelse(famsup=='no', 0,1))
stmath$paid = as.factor(ifelse(paid=='no', 0,1))
stmath$activities = as.factor(ifelse(activities=='no', 0,1))
stmath$higher = as.factor(ifelse(higher=='no', 0,1))

stmath$address = as.factor(ifelse(address=='R', 0,1))
stmath$famsize = as.factor(ifelse(famsize=='LE3', 0,1))
stmath$Pstatus = as.factor(ifelse(Pstatus=='A', 0,1))

## kNN

## Normalizing our data
normalize = function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

LESSnon_num_var = stmath[, c(1,3,4,5,9,10,11,12,13)]
LESSnum_var = stmath[, c(2,6,7,8,14,15,16,17,18)]

LESSPnorm = as.data.frame(lapply(LESSnum_var, normalize))
LESSPnormed=cbind(LESSPnorm, LESSnon_num_var)
names(LESSPnormed)

# Creating testing and training data sets

library(caret) 
set.seed(14)
LESSstratid=createDataPartition(LESSPnormed$paid, p=0.75, list=FALSE) ## Create the partition again setting 75% to training
LESSNormstmath_train=LESSPnormed[LESSstratid,] ## Creating the Training dataset
LESSNormstmath_test=LESSPnormed[-LESSstratid,] ## Creating the test data set

## Run the KNN algorithm on the Training data set

library(class)

LESSknn_mod = knn(train=LESSNormstmath_train[,-16], test=LESSNormstmath_test[,-16], cl=LESSNormstmath_train[,16], k=16)

## Test model predictions using the confusion matrix

library(e1071)
confusionMatrix(as.factor(LESSknn_mod),as.factor(LESSNormstmath_test[,16]), 
                positive="1")

## KNN with cross validation
LESSfolds=createFolds(LESSPnormed$paid,k=10)

LESSknn_cvaccuracy = sapply(LESSfolds, function(x){
  trainset=LESSPnormed[-x,]
  testset=LESSPnormed[x,]
  model=knn(trainset[,-16], test=testset[,-16], cl=trainset[,16], k=16)
  obj=confusionMatrix(as.factor(model),as.factor(testset[,16]), 
                      positive="1")
  metrics=c(obj$overall[1], obj$overall[2], obj$byClass[1], obj$byClass[2])
  return(metrics)
})

LESSknn_cvaccuracy

### Naive Bayes Classifier

library(e1071)

## Holdout sampling apporach

LESSNormstmath_train_nb=LESSNormstmath_train[,-16] ## Pulling the features out from the train set
LESSNormstmath_test_nb=LESSNormstmath_test[,-16] ## Pulling the features set out of the test set

LESSnbmath=naiveBayes(LESSNormstmath_train_nb, as.factor(LESSNormstmath_train[,16])) ## Run the Naive Bayes model

LESSnbpred=predict(LESSnbmath, LESSNormstmath_test_nb, type="class") ## Generate Test Predictions


## Generate the confusion matrix

confusionMatrix(as.factor(LESSnbpred),as.factor(LESSNormstmath_test[,16]), 
                positive="1")

## Naive Bayes with cross validation
LESSnb_folds=createFolds(LESSPnormed$paid,k=10)

LESSnb_cvaccuracy = sapply(LESSfolds, function(x){
  trainset=LESSPnormed[-x,]
  testset=LESSPnormed[x,]
  
  LESStrainx=trainset[,-16]
  LESStestx=testset[,-16]
  
  model=naiveBayes(LESStrainx, as.factor(trainset[,16]))
  npred=predict(model, LESStestx, type="class")
  
  obj=confusionMatrix(as.factor(npred),as.factor(testset[,16]), 
                      positive="1")
  metrics=c(obj$overall[1], obj$overall[2], obj$byClass[1], obj$byClass[2])
  return(metrics)
})

LESSnb_cvaccuracy

## End of kNN and Naive Bayes section.