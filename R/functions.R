library(DMwR)

# calculate the confusion matrix and print out
# the correct predicted defaults
confuMatrix <- function(pr, y_true, thred) {
  y_pred = ifelse(pr > thred, 1, 0)
  cm = table(y_true, y_pred )
#   print(cm)
  print(1/(1+(cm[2,1] + cm[1,2])/(2*cm[2,2])))
  print(cm[2,2] - cm[2,1] - cm[1,2])
}

# ===================================================
# given training and testing sets, predict default on
# testing set
glm.classifier <- function(model, train, test) {
  
  clf= model(factor(train[, ncol(train)]) ~ ., data = train[ , -ncol(train)], family = binomial)
  pr = predict(clf, test, type="response")
  
  return(pr)
}

# ===================================================
fitting01 <- function(model, train, test) {
  
  clf= model(factor(train[, ncol(train)]) ~ ., data = train[ , -ncol(train)], family = binomial)
  pr = predict(clf, test, type="response")
  
  return(pr)
}

# ===================================================
# calculate the mean absolute value
MAE <- function(x,y) {
  diff <- x-y
  val <- mean(abs(diff))
  return(val)
}

round.loss <- function(x) {
  x <- floor(x)
  x[which(x>100) ] <- 100
  x[which(x<1) ] <- 0
  return(x)
}


transform <- function(x, lambda) {
  new_x <- (x^lambda - 1)/lambda
  return(new_x)
}

filter_non_default <- function(data) {
  data$f1 <- data$f528-data$f527
  data$f1 <- transform(data$f1, 0.07)
  dx <- data[which(data$loss==1), ]
  second_min <- sort(dx$f1)[2]
  
  idz <- which(data$f1>=second_min)
  return(idz)
}
# =============================================================
norm01 <- function(data) {
  nl <- ncol(data)
  mean_i <- lapply(1:nl, function(i) {mean(data[, i])})
  std_i <- lapply(1:nl, function(i) {sd(data[, i])})
  new.data <- (data-mean_i)/std_i
    
  return(new.data)
}

#preProcValues <- preProcess(training, method = c("center", "scale"))
# =============================================================
get.data.01 <- function(data, have_loss) {
  clean.data <- centralImputation(data)
  if (have_loss) {
    nl <- ncol(clean.data)
    #clean.data[, -nl] <- transform(clean.data[, -nl], -0.0035)
    clean.data[, -nl] <- norm01(clean.data[, -nl])
    clean.data$loss <- ifelse(clean.data$loss > 0, 1, 0)    
  } else {
    #clean.data <- transform(clean.data, -0.0035)
    clean.data <- norm01(clean.data)
  }
  
  return(clean.data)
}

# =============================================================
Fscore <- function(X, indix) {
  # calculate the 
  # given tne columns and indices of defaults
  X_def = X[indix]
  X_nondef = X[-indix]
  mean_X = mean(X, na.rm = TRUE)
  mean_def = mean(X_def, na.rm = TRUE)
  mean_nondef = mean(X_nondef, na.rm = TRUE)
  
  numerator = (mean_def - mean_X)**2 + (mean_nondef - mean_X)**2
  denominator = var(X_def, na.rm = TRUE) + var(X_nondef, na.rm = TRUE)
  
  return(numerator/denominator)
}

# =============================================================
filter_feature <- function(data) {
  indix <- which(data$loss > 0)

  vec <- sapply(seq(2,ncol(data)-1), function(i) { Fscore(data[, i], indix)})
  return(vec)
}


# =============================================================
# choose_best_feature <- function(data, thred) {
#   features <- filter_feature(data)
#   best_features <- which(features > thred)
#   
#   sprintf("At threshold %f there are %d features.", thred, best_features)
#   
#   best_features <- c(best_features +1, ncol(data))
#   
#   new_data <- data[, best_features]
#   new_data <- centralImputation(new_data)
#   
#   return(new_data)
# }

choose_best_feature <- function(data, thred1, thred2) {
  features <- filter_feature(data)
  best_features <- which(features > thred1 & features <= thred2)
  
  sprintf("At threshold %f there are %d features.", thred1, best_features)
  
  best_features <- c(best_features +1, ncol(data))
  
  new_data <- data[, best_features]
  new_data <- centralImputation(new_data)
  
  return(new_data)
}