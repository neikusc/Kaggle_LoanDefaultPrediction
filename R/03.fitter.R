library(quantreg)

fitter <- function(dataTrain, dataTest, idz) {
  features <- c("f281","f67","f630","f230","f404","f2","f77","f514", "f670","f596",
                "f376","f68","f479","f261","f588","f734","f621","f654","f410","f73",
                "f291","f696","f158","f426","f739","f617")
  idx <- which(dataTrain$loss>0)
  raw.train <- dataTrain[idx, c(features,"loss")]
  raw.train <- centralImputation(raw.train)
  nl = ncol(raw.train)-1
  mean_i <- lapply(1:nl, function(i) {mean(raw.train[, i])})
  std_i <- lapply(1:nl, function(i) {sd(raw.train[, i])})
  train <- (raw.train[,-ncol(raw.train)]-mean_i)/std_i 
  train$loss <- raw.train$loss
  
  
  raw.test <- dataTest[idz , features]
  raw.test <- centralImputation(raw.test)
  test <- (raw.test-mean_i)/std_i
  
  
  model <- rq(loss~., data=train)
  y_pred <- predict(model, newdata=test, type="response")  
  y_pred <- round.loss(y_pred)
  
  loss_pred <- rep(0, nrow(dataTest))
  loss_pred[idz] <- y_pred
  return(loss_pred)
}