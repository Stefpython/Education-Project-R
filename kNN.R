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

## Test model predictions using the confusion matrix

library(e1071)
confusionMatrix(as.factor(knnmodel),as.factor(Ploan_test[,14]), 
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


