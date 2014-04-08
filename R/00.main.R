setwd('/home/neik/GitHub/Kaggle_LoanDefaultPrediction/R')
source("functions.R")
source("02.classifier.R")
source("03.fitter.R")
library(caTools)
library(caret)
set.seed(2014)

# load "data" dataframe
load("train_v2.Rda") 
# load "dataTest" dataframe
load("test_v2.Rda")

# classifying for the id of defaults
idz <- classifier(data, dataTest, 0.70)
length(idz)/nrow(dataTest)

# GDF: fit the loss
pred_loss <- fitter(data, dataTest, idz)

hist(data$loss, col=rgb(1, 0, 0,0.5),main="Training Set", ylim=c(0,1500), breaks=101,
     xlab ="Loss", ylab ="Frequency")
hist(pred_loss, col=rgb(0, 1, 0,0.5), main="Testing Set", ylim=c(0,1500), breaks=101, add=T)


# should be: 175843 on total 210944 samples
sum(pred_loss)/175843

# ==================================================================================
# Save file for submission
# ==================================================================================
newdata = data.frame(id = dataTest$id , loss = pred_loss) 
write.csv(newdata, file = "submission.csv", row.names = FALSE, quote=FALSE)
