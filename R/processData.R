setwd('/home/neik/GitHub/Kaggle_LoanDefaultPrediction/R')
source("functions.R")


# read trainning sets and save it in text file for quick process
data = read.csv('train_v2.csv')
save(data, file = "train_v2.Rda")

dataTest = read.csv('test_v2.csv')
save(dataTest, file = "test_v2.Rda")
