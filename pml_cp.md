# Practical Machine Learning Course Project
Guillermo Pachón  
November 10, 2016  



## 1. Summary

### 1.1 Background

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how *much* of a particular activity they do, but they rarely quantify *how well they do it*. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### 1.2 Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from Groupware Technologies - Human Activity Recognition, Weight Lifting Exercises Dataset: http://groupware.les.inf.puc-rio.br/har

### 1.3 Project

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. I may use any of the other variables to predict with.

The following report will describe how built the model, how used cross validation, what think the expected out of sample error is, and why I made the choices I did. I will also use the prediction model to predict 20 different test cases.

## Exploring the Data

We need to load required packages and set parallel options for improved performance.


```r
# Required packages
library(RCurl); library(caret); library(relaxo); library(parallel); library(doParallel); library(reshape2);
```

```
## Loading required package: bitops
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lars
```

```
## Loaded lars 1.2
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
# Set parallel options
cluster <- makeCluster(detectCores() - 1) # Leave 1 for OS
registerDoParallel(cluster)
```

Load the data from the provided files. 


```r
training <- read.csv(file="pml-training.csv", na.strings = c("NA", "#DIV/0!"))
testing <- read.csv(file="pml-testing.csv", na.strings = c("NA", "#DIV/0!"))

paste("TRAINING: Rows: ", dim(training)[1], ". Columns: ", dim(training)[2], ".", sep = "")
```

```
## [1] "TRAINING: Rows: 19622. Columns: 160."
```

Now we will do some exploration and make some analisys. Using the information in *training* we will create the training and test data to test some prediction methods.

But, before the analisys, some cleaning work sholud be made in training data. Several columns contains only NA values making nearly imposible to validate the methods with the training data as is. We will:

* Remove covariates with more than 80% missing values.
* Remove near zero covariates.


```r
# Columns contains only NA values, so that columns will be removed
training.mNA <- sapply(colnames(training), function(x) if(sum(is.na(training[, x])) > 0.8*nrow(training)){return(T)}else{return(F)})
training <- training[, !training.mNA]

# Create partitions for train and test
set.seed(83538)
inTrain <- createDataPartition(training[,1], p = 0.5, list = FALSE)
trainDF <- training[inTrain,]
testDF <- training[-inTrain,]
```

## Fitting Models

To make an automated analisys, create a function to test some methods and try to identify the one that gets best results (Accuracy). 


```r
# Function testModel return the Accuracy from the confusionMatrix.
testModel <- function(tr, ts, m = "lm", usePCA = FALSE) {
  preProc = NULL
  if (usePCA) { preProc = "pca" }
  mFit <- train(classe ~ ., method = m, data = tr, preProcess = preProc, trControl = fitControl)
  cMat <- confusionMatrix(ts$classe, predict(mFit, newdata = ts))
  Accuracy = round(cMat$overall[[1]], 6)
  Accuracy
}
```

In all cases, test the classiﬁer with 10-fold cross-validation.


```r
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
```

Test the following models:

* CART
* Random Forest
* Stochastic Gradient Boosting
* Naive Bayes

For every method tested, the testModel funcion is called, first without and then with a Principal Components Analisys.



From the analisys we get the following numbers:


```
##   Method Accuracy   PCA
## 1  rpart 0.661672 FALSE
## 2     rf 0.999796 FALSE
## 3    gbm 0.999694 FALSE
## 4     nb  0.86157 FALSE
## 5  rpart 0.285015  TRUE
## 6     rf 0.982977  TRUE
## 7    gbm  0.93792  TRUE
## 8     nb  0.81998  TRUE
```

## Model Selection

From the model analisys we get that the best method to estimate the outcome is Random Forest (Accuracy: 0.999796) or Stochastic Gradient Boosting (Accuracy: 0.999694) so we will continue with **Random Forest**.

Now using the model selected, use the leave-one-subject-out test in order to measure whether our classiﬁer trained for some subjects is still useful for a new subject.


```r
# Partition the data by user_name to leave-one-subject-out test
unLvls <- levels(training$user_name)
# Training sets removing one user
trnUN1 <- training[training$user_name != unLvls[1], ]
trnUN2 <- training[training$user_name != unLvls[2], ]
trnUN3 <- training[training$user_name != unLvls[3], ]
trnUN4 <- training[training$user_name != unLvls[4], ]
trnUN5 <- training[training$user_name != unLvls[5], ]
trnUN6 <- training[training$user_name != unLvls[6], ]
# Tesing sets using the user removed in training
tstUN1 <- training[training$user_name == unLvls[1], ]
tstUN2 <- training[training$user_name == unLvls[2], ]
tstUN3 <- training[training$user_name == unLvls[3], ]
tstUN4 <- training[training$user_name == unLvls[4], ]
tstUN5 <- training[training$user_name == unLvls[5], ]
tstUN6 <- training[training$user_name == unLvls[6], ]

# Generate models for train data leaving one user at a time
leaveSbj1Fit <- train(classe ~ ., method = "rf", data = trnUN1, trControl = fitControl)
leaveSbj2Fit <- train(classe ~ ., method = "rf", data = trnUN2, trControl = fitControl)
leaveSbj3Fit <- train(classe ~ ., method = "rf", data = trnUN3, trControl = fitControl)
leaveSbj4Fit <- train(classe ~ ., method = "rf", data = trnUN4, trControl = fitControl)
leaveSbj5Fit <- train(classe ~ ., method = "rf", data = trnUN5, trControl = fitControl)
leaveSbj6Fit <- train(classe ~ ., method = "rf", data = trnUN6, trControl = fitControl)
# Generate confusion matrix for each model
leaveSbj1ConfcMat <- confusionMatrix(tstUN1$classe, predict(leaveSbj1Fit, newdata = tstUN1))
leaveSbj2ConfcMat <- confusionMatrix(tstUN2$classe, predict(leaveSbj2Fit, newdata = tstUN2))
leaveSbj3ConfcMat <- confusionMatrix(tstUN3$classe, predict(leaveSbj3Fit, newdata = tstUN3))
leaveSbj4ConfcMat <- confusionMatrix(tstUN4$classe, predict(leaveSbj4Fit, newdata = tstUN4))
leaveSbj5ConfcMat <- confusionMatrix(tstUN5$classe, predict(leaveSbj5Fit, newdata = tstUN5))
leaveSbj6ConfcMat <- confusionMatrix(tstUN6$classe, predict(leaveSbj6Fit, newdata = tstUN6))
# Generate accuracy matrix for each model
lvSbj1AccTb <- leaveSbj1ConfcMat$table / summary(tstUN1$classe)
lvSbj2AccTb <- leaveSbj2ConfcMat$table / summary(tstUN2$classe)
lvSbj3AccTb <- leaveSbj3ConfcMat$table / summary(tstUN3$classe)
lvSbj4AccTb <- leaveSbj4ConfcMat$table / summary(tstUN4$classe)
lvSbj5AccTb <- leaveSbj5ConfcMat$table / summary(tstUN5$classe)
lvSbj6AccTb <- leaveSbj6ConfcMat$table / summary(tstUN6$classe)
# Generate summed and averaged accuracy matrix
listMatrix <- list(lvSbj1AccTb, lvSbj2AccTb, lvSbj3AccTb, lvSbj4AccTb, lvSbj5AccTb, lvSbj6AccTb)
averageMatrix <- Reduce('+', listMatrix) / 6
```

The results are clear in the following plot:


```r
# Transform matrix into data.frame
avgM <- melt(averageMatrix)
# Plot results
g <- ggplot(avgM, aes(Reference, Prediction)) + labs(title = "Averaged Confission Matrix for leave-one-out-test")
g <- g + geom_tile(aes(fill = value), colour = "white")
g <- g + geom_text(aes(label= ifelse(value == 0, "", round(value, 4))), color = "black", size = 4)
g <- g + scale_fill_gradient(low = "white", high = "steelblue")
g
```

![](pml_cp_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
**Figure**. *Averaged Confission Matrix for leave-one-out-test"*.

## Prediction

Now, after verifiying the performance of the model selected, predict the *classe* for the **training** data.


```r
mFit <- train(classe ~ ., method = "rf", data = training, trControl = fitControl)
# final model
mFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 41
## 
##         OOB estimate of  error rate: 0.01%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5580    0    0    0    0 0.0000000000
## B    1 3796    0    0    0 0.0002633658
## C    0    1 3421    0    0 0.0002922268
## D    0    0    0 3216    0 0.0000000000
## E    0    0    0    0 3607 0.0000000000
```

```r
# prediction
prediction <- predict(mFit, testing)

# De-register parallel processing cluster
stopCluster(cluster)
```

So, the predicted 20 values por testing are: ``A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A``.

## Conclussions

* We use Random Forests as prediction method with 10-fold cross-validation. This method give us a 0.999796 Accuracy.

* The final random forests model contains 500 trees with 41 variables tried at each split.

* Estimated out of sample error rate for the random forests model is 0.01% as reported by the final model.

* Also to test the efectivity with new subjects we use the Leave One Out Cross Validation (training and test sets leaving one subject in the test). This test gave the following Accuracy by class: (A) 99.7%, (B) 90.12%, (C) 88.51%, (D) 97.38%, (E) 99.63%.

* The execution of the train function for the prediction methods is very highly procesing consuming, the previus analisys took almost two hours to complete (using parallel options for improved performance).
