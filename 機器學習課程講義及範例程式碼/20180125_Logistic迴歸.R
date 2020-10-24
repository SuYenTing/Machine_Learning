# Logistic迴歸模型範例
rm(list=ls());gc()
library(nnet)

# 讀取鳶尾花資料
data <- iris

# 以隨機抽樣方式建立訓練組(70%)與預測組樣本(30%)
set.seed(10)
trainIndex <- createDataPartition(data$Species, p = 0.7, list = FALSE)
irisTrain <- data[trainIndex,]     # 訓練組樣本
irisTest  <- data[-trainIndex,]    # 預測組樣本

# 以訓練集資料做Logistic模型
model <- multinom(Species ~ . , data=irisTrain)

# 模型預測
predict <- predict(model, irisTest)

# 建立混淆矩陣
confusionMatrix <- table(irisTest$Species, predict, dnn=c("實際","預測"))

# 計算模型準確率
accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix)


