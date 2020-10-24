# 類神經網路模型範例
rm(list=ls());gc()
library(dplyr)    # 整理資料套件
library(reshape)  # 整理資料套件
library(caret)    # 機器學習套件，用於尋找最佳參數
library(nnet)     # 類神經網路模型套件
library(devtools) # 從github下載plot.nnet()函數以方便繪圖

# 引用github的plot.nnet()套件
source_url(paste0("https://gist.githubusercontent.com/fawda123/7471137",
                  "/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r"))

# 讀取鳶尾花資料
data <- iris 

# 以隨機抽樣方式建立訓練組(80%)與預測組樣本(20%)
set.seed(5)
trainDataRatio <- 0.8
trainIndex <- sample(c(1:nrow(data)), nrow(data)*trainDataRatio)
trainData <- data[trainIndex,]     # 訓練特徵
testData  <- data[-trainIndex,]    # 預測特徵

# 參數組合
size <- seq(from = 1, to = 4, by = 1)             # 隱藏層神經元數目
decay <- c(0, 0.01, 0.1, 0.5)                     # 權重衰變參數
nnetTune <- expand.grid(size=size, decay=decay)

# 尋找最佳參數組合
tuneModel <- train(Species ~ ., data = trainData, method = "nnet", tuneGrid = nnetTune)          

# 取出最佳參數組合模型
bestParm <- tuneModel$bestTune

# 訓練類神經網路模型
model <- nnet(formula = Species ~ ., linout = T, size = bestParm$size, 
              decay = bestParm$decay, data = trainData)

# 繪製類神經網路模型圖
plot.nnet(model, wts.only = F) 

# 預測結果
pred_result <- predict(model, testData, type = 'class')

# 實際/預測分類矩陣
confusionMatrix <- table(pred_result,testData$Species)


