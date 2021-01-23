### Group Project: EDUCATION IN PORTUGUESE SCHOOLS

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

  
################### Tree to find out most relevant variables

names(stmath)

#Creating a sample and training dataset
library(caret)
library(tree)
set.seed(01)
stratid=createDataPartition(stmath$schoolsup, p=0.75, list=F) ## Create the data partition
stmathtree_train=stmath[stratid,] ### Create the training data set
stmathtree_test=stmath[-stratid,] ## Create the testing data set

set.seed(02)
tree.stmath=tree(schoolsup~., 
             data=stmathtree_train) ## Estimates the classification tree

summary(tree.stmath)
#The misclassification reported is the entropy and it is aournd 7%
plot(tree.stmath)
text(tree.stmath, pretty=0)
#Most important variables: age+health+G3+studytime+goout+Fedu+Medu+health+Fjob+Walc+famrel+internet+famsup+famsize+failures+paid

#Test Error
tree.stmath.pred=predict(tree.stmath, stmathtree_test, 
                     type="class") 
library(e1071)
conftabletree = confusionMatrix(as.factor(tree.stmath.pred),as.factor(stmathtree_test$schoolsup), 
                              positive="1")
conftabletree

accuracy_tree = mean(tree.stmath.pred == stmathtree_test$schoolsup)
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
bag.stmath = randomForest(schoolsup~., data=stmathtree_train, 
                    mtry=(ncol(stmathtree_train)-1),importance=T,
                    ntree=500)
bag.stmath
bag.stmath.pred = predict(bag.stmath, stmathtree_test, type = 'class')
conftabletree = confusionMatrix(as.factor(bag.stmath.pred),as.factor(stmathtree_test$schoolsup), 
                                positive="1")

accuracy_bag = mean(bag.stmath.pred == stmathtree_test$schoolsup)
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
#Most important variables: Age+G3+Fjob+address+Mjob+health+freetime+famrel+absences+Medu+paid+goout+Walc+studytime+internet+sex+romatic+Fedu+school+activities+failures+Pstatus

#Comparing Tree and Bagging
comp_tree_bag = rbind(modelfitmetrics_stmathtree,modelfitmetrics_stmathbag)
comp_tree_bag

#RandomForest still needs to be done. For that, set mtry = (ncol(stmathtree_train)-1) in the bagging formula



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

#Deriving linear models from tree and bag. Tree should perform better
stmath = as.data.frame(stmath)
mod1_lm = lm(schoolsup~age+health+G3+studytime+goout+Fedu+Medu+health+Fjob+Walc+famrel+internet+famsup+famsize+failures+paid, data = stmath)
summary(mod1_lm)

mod2_lm = lm(schoolsup~., data = stmath)
summary(mod2_lm)

regmetrics=function(model,data, y)
{
  p=predict(model,newdata=data)
  MAPE=sum((abs(y-p))/y)/nrow(data)
  MAD=sum((y-p)^2)/nrow(data)
  MSE=sum(abs(y-p))/nrow(data)
  x = cbind(MAPE,MAD,MSE)
  return (x)
}

set.seed(04)
mod1_folds=createFolds(stmath$schoolsup,k=10)

regmetkfold_mod1_lm = sapply(mod1_folds, function(x){
  trainset=stmath[-x,]
  testset=stmath[x,]
  model=lm(schoolsup~age+health+G3+studytime+goout+Fedu+Medu+health+Fjob+Walc+famrel+internet+famsup+famsize+failures+paid, data = trainset)
  msp=regmetrics(model, testset, testset$schoolsup)
  names(msp) = c('MAPE', 'MAD', 'MSE')
  return(msp)
})

regmetkfold_mod1_lm
regmetkfold_mod1_lm = apply(regmetkfold_mod1_lm,1,mean)
regmetkfold_mod1_lm


