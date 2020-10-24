# 支持向量機模型範例
rm(list=ls());gc()
library(e1071)    # Libsvm函數套件
library(dplyr)    # 資料整理套件
library(ggplot2)  # 繪圖套件 

# 讀取鳶尾花資料
data <- iris %>% filter(Species!="setosa") %>% select(-Species) 
class <- iris %>% filter(Species!="setosa") %>% select(Species) %>% droplevels()

# 以隨機抽樣方式建立訓練組(80%)與預測組樣本(20%)
set.seed(5)
trainDataRatio <- 0.8
trainIndex <- sample(c(1:nrow(data)), nrow(data)*trainDataRatio)
trainData <- data[trainIndex,]     # 訓練特徵
trainClass <- class[trainIndex,]   # 訓練分類
testData  <- data[-trainIndex,]    # 預測特徵
testClass  <- class[-trainIndex,]  # 預測分類

# 參數挑選範圍
costRange <- c(2^(-8:8))
gammaRange <- c(2^(-8:8))

# 回測參數組合
paraList <- expand.grid(costRange=costRange, gammaRange=gammaRange)

# 交叉驗證函數 
FindBestPara <- function(ix){
  cost <- paraList[ix,1]
  gamma <- paraList[ix,2] 
  model <- svm(trainData, trainClass, cross=10, kernel="radial", cost=cost, gamma=gamma)
  return(model$tot.accuracy)
}

# 交叉驗證求參數
cvResult <- sapply(c(1:nrow(paraList)), FindBestPara)

# 繪製各參數組合的等高線函數圖
plotData <- paraList %>%
  transmute(cost=log(costRange,2), gamma=log(gammaRange,2)) %>%
  bind_cols(tibble(acc=cvResult))

ggplot(plotData, aes(x = cost, y = gamma, z = acc)) +
  theme_bw() +
  stat_contour(aes(colour = ..level..), binwidth = 1, size=1.2) + 
  scale_colour_gradientn(colours=rev(rainbow(10))) +
  xlab("log(Cost,2)") +
  ylab("log(Gamma,2)") +
  scale_fill_manual(name="Accuracy")

# 依交叉驗證結果挑最好參數組數
bestCV <- max(cvResult)
bestParaSite <- which(cvResult==bestCV)[1]
bestCost <- paraList[bestParaSite,1]
bestGamma <- paraList[bestParaSite,2]

# 訓練模型
model <- svm(trainData, trainClass, cost=bestCost, gamma=bestGamma)

# 模型訓練結果
summary(model)

# 預測結果
pred_result <- predict(model, testData)

# 實際/預測分類矩陣
confusionMatrix <- table(pred_result,testClass)

# 預測準確率
accurRate <- sum(diag(confusionMatrix))/length(testClass)




