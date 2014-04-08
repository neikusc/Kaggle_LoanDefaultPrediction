setwd('/home/neik/GitHub/Kaggle_LoanDefaultPrediction/R')
source("functions.R")
source("02.classifier.R")
source("03.fitter.R")
library(caTools)
library(caret)

set.seed(2014)

# load dataTrain
load("train_v2.Rda")

id <- sample(nrow(data), round(nrow(data)*0.67))
dataTrain <- data[id, ]
dataTest <- data[-id, ]
true_loss <- dataTest$loss

# classifying for the id of defaults
idz <- classifier(dataTrain, dataTest, 0.70)
length(idz)/nrow(dataTest)
length(idz)/length(which(dataTest$loss>0))

# fit loss
pred_loss <- fitter(dataTrain, dataTest, idz)

MAE(pred_loss, true_loss)
y_zero <- rep(0, length(true_loss))
MAE(y_zero, true_loss)

par(mfrow = c(1,2))
hist(true_loss, col = "blue", ylim=c(0,900), main="True Loss", 
     xlim=c(0,101), xlab = "Loss", ylab = "Frequency", breaks=101)
hist(pred_loss, col = 'green', ylim=c(0,900), main="Predicted Loss",
     xlim=c(0,101), xlab = "Loss", ylab = "Frequency", breaks=101)

sum(pred_loss)/sum(true_loss)
