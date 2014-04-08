library(DMwR)
# library(gbm)
library(foreach)
library(randomForest)

classifier <- function(dataTrain, dataTest, thred) {
  xTrain <- dataTrain[, c("f2","f222","f271","f274","f527","f528","f777", "loss")]
  xTrain <- centralImputation(xTrain)
  raw.train <- xTrain[, c("f2","f222","f777")]
  raw.train$f528_f274 <- xTrain$f528 - xTrain$f274
  raw.train$f528_f527 <- xTrain$f528 - xTrain$f527
  raw.train$logf271 <- log(xTrain$f271+1)
  raw.train$loss <- ifelse(xTrain$loss>0,1,0)

  xTest <- dataTest[, c("f2","f222","f271","f274","f527","f528","f777")]
  xTest <- centralImputation(xTest)
  raw.test <- xTest[ , c("f2","f222","f777")]
  raw.test$f528_f274 <- xTest$f528 - xTest$f274
  raw.test$f528_f527 <- xTest$f528 - xTest$f527
  raw.test$logf271 <- log(xTest$f271+1)
  
  train <- raw.train[sample(nrow(raw.train)), ]
  
#   clf <- gbm(loss ~ ., data=train, distribution="adaboost",
#               n.trees=100, shrinkage=.01, cv.folds=10,
#               train.fraction=.9, verbose = FALSE , n.cores=4)
  clf <- foreach(ntree=rep(125, 4), .combine=combine) %do%
    randomForest(loss~., data=raw.train, ntree=ntree, ntry=10)
  
  pr <- predict(clf, newdata = raw.test, type="response")

  # classify the defaults  
  default <- ifelse(pr > thred, 1, 0)
  id_default <- which(default>0)

# # print out the best threshold
#   y_true <- ifelse(dataTest$loss>0,1,0)
#   confuMatrix(pr, y_true, thred)
#   cat('AUC:', colAUC(pr, y_true),'\n')
#   for (x in seq(0.50,0.64,by=0.01) ) {
#     confuMatrix(pr, y_true, x)  
#   }
  
  return(id_default)
}