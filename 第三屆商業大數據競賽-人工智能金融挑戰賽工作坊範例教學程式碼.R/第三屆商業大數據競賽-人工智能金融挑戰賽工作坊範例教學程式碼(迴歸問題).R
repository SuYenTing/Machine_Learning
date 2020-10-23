# 20200926_第三屆商業大數據競賽-人工智能金融挑戰賽工作坊範例教學程式碼(迴歸問題)
# 清空資料及記憶體
rm(list = ls());gc()

# 載入套件
library(tidyverse)        # 整理資料套件
library(readxl)           # 讀取excel檔案套件
options(scipen = 999)     # 數字不要以科學記號呈現

# 設定檔案載入位置 請自行設定檔案路徑
filePath <- "C:/Users/adm/Desktop/人工智能金融挑戰賽工作坊範例教學程式碼"
setwd(filePath)

################# 載入檔案 #################
# 載入資料
rawData <- read_excel("Real estate valuation data set.xlsx")

# 重新命名欄位
colnames(rawData)
colnames(rawData) <- c("No", "transDate", "houseAge", "distanceMRT", "convenienceStoreNums",
                       "latitude", "longitude", "y")


################# 整理資料 #################
# 觀察資料
str(rawData)
# 繪製直方圖
for(ix in c(1:ncol(rawData))) hist(rawData %>% pull(ix), main = colnames(rawData)[ix])

# 刪除不需要的欄位
rawData <- rawData %>% select(-c("No", "transDate"))

# 載入相關係數圖的套件
library(corrplot)

# 繪製相關係數圖
corPlot <- cor(rawData, use = "pairwise.complete.obs")
corrplot(corPlot, 
         method = "color", 
         type = "lower",
         order = "hclust", 
         addCoef.col = "black",
         tl.col = "black", 
         tl.srt = 20,
         diag = FALSE) 

# 隨機切分70/30訓練集與測試集資料
set.seed(1234)
rndNums <- sample(c(1:nrow(rawData)), size = round(nrow(rawData)*0.7), replace = F)
trainData <- rawData[rndNums, ]
testData <- rawData[-rndNums, ]

# 觀察資料維度
dim(trainData)
dim(testData)

# 進行特徵標準化
for(iy in c(1:ncol(trainData))){
  
  # 取出訓練期及測試期特徵資料
  trainFeature <- trainData %>% pull(iy)
  testFeature <- testData %>% pull(iy)
  
  # 計算特徵平均值及標準差
  trainFeatureMean <- mean(trainFeature)  # 計算特徵平均值
  trainFeatureSd <- sd(trainFeature)      # 計算特徵標準差
  
  # Z分數
  trainFeature <- (trainFeature-trainFeatureMean)/trainFeatureSd
  testFeature <- (testFeature-trainFeatureMean)/trainFeatureSd
  
  # 壓縮離群值(大於正負3倍標準差)
  trainFeature[which(trainFeature > 3)] <- 3
  trainFeature[which(trainFeature < (-3))] <- (-3)
  testFeature[which(testFeature > 3)] <- 3
  testFeature[which(testFeature < (-3))] <- (-3)
  
  # 將整理好特徵取代回原資料
  trainData[, iy] <- trainFeature
  testData[, iy] <- testFeature
}

# 繪製特徵處理過後的直方圖
for(ix in c(1:ncol(trainData))) hist(trainData %>% pull(ix), main = colnames(trainData)[ix])
for(ix in c(1:ncol(testData))) hist(testData %>% pull(ix), main = colnames(testData)[ix])

# 將訓練及測試集資料另外儲存 因為不同的模型所需要的Y和X格式會不相同
str(trainData)
str(testData)
originTrainData <- trainData
originTestData <- testData

# 建立MSE函數
MSE <- function(y_real, y_pred) mean((y_real-y_pred)^2)


################# 決策樹模型(Decision Tree) #################
# 匯入決策樹套件
library(rpart)            # 決策樹套件
library(partykit)         # 決策樹繪圖套件

# 載入資料
trainData <- originTrainData
testData <- originTestData

# 決策樹參數設定
rpartConfigure <- rpart.control(minsplit = 10,
                                minbucket = round(10/3),
                                cp = 0.0001,
                                xval = 10,
                                maxdepth = 5)
# 決策樹參數說明
# minsplit: 每個節點上至少要有多少樣本才會進行拆分
# minbucket: 在末端的節點上最少要幾有幾筆樣本
# cp: 複雜度參數(complexity parameter) 每次拆分節點時需要超過這個值才能拆分 值愈大模型學習愈保守
# xval: 交叉驗證折數
# maxdepth: 樹的深度

# 執行決策樹預測類型為分類
treeModel <- rpart(formula = y ~ ., 
                   data = trainData,
                   method = "anova",
                   control = rpartConfigure)

# 模型訓練集預測
predictTrainResult <- predict(treeModel, trainData)

# 計算訓練集準確度
treeModelTrainMSE <- MSE(y_real = trainData$y, y_pred = predictTrainResult)

# 模型訓練集預測
predictTestResult <- predict(treeModel, testData)

# 計算訓練集準確度
treeModelTestMSE <- MSE(y_real = testData$y, y_pred = predictTestResult)

# 視覺化交叉驗證後結果
plotcp(treeModel)

# 繪製決策樹
plot(as.party(treeModel))
# rpart.plot(cartModel)

# 進行剪枝(選取交叉驗證錯誤率(xerror)最小的cp)
bestCp <- treeModel$cptable[which.min(treeModel$cptable[, "xerror"]), "CP"]
prunedTreeModel <- prune(treeModel, cp = bestCp)

# 視覺化剪枝結果
plotcp(prunedTreeModel)

# 繪製剪枝後決策樹
plot(as.party(prunedTreeModel))
# rpart.plot(prunedTreeModel)

# 觀看剪枝後決策樹重要變數
attr(prunedTreeModel$variable.importance, "names")

# 模型訓練集預測
predictTrainResult <- predict(prunedTreeModel, trainData)

# 計算訓練集準確度
prunedTreeModelTrainMSE <- MSE(y_real = trainData$y, y_pred = predictTrainResult)

# 模型訓練集預測
predictTestResult <- predict(prunedTreeModel, testData)

# 計算訓練集準確度
prunedTreeModelTestMSE <- MSE(y_real = testData$y, y_pred = predictTestResult)

# 準確度比較
treeModelTrainMSE           # 剪枝前訓練期預測準確度
treeModelTestMSE            # 剪枝前測試期預測準確度
prunedTreeModelTrainMSE     # 剪枝後訓練期預測準確度
prunedTreeModelTestMSE      # 剪枝後測試期預測準確度


################# 隨機森林模型(Random Forest) #################
# 載入套件
library(randomForest)

# 載入資料
trainData <- originTrainData
testData <- originTestData

# 隨機森林模型
rfModel <- randomForest(y~., 
                        data = trainData, 
                        importance = T, 
                        ntree = 100)

# 查看隨機森林結果
rfModel

# 繪製隨機森林樹模型
plot(rfModel, lwd = 2)

# 觀看隨機森林的重要特徵
importance(rfModel)

# 模型訓練集預測
predictTrainResult <- predict(rfModel, trainData)

# 計算訓練集準確度
rfModelTrainMSE <- MSE(y_real = trainData$y, y_pred = predictTrainResult)

# 模型訓練集預測
predictTestResult <- predict(rfModel, testData)

# 計算訓練集準確度
rfModelTestMSE <- MSE(y_real = testData$y, y_pred = predictTestResult)


################# 極限梯度提升模型(XGBoost) #################
library(xgboost)

# 載入資料
trainData <- originTrainData
testData <- originTestData

# 整理訓練及測試集資料
trainLabel <- trainData %>% pull(y)
trainData <- trainData %>% select(-y) %>% data.matrix()  # 此處將所有的類別資料轉為數值
testLabel <- testData %>% pull(y)
testData <- testData %>% select(-y) %>% data.matrix()    # 此處將所有的類別資料轉為數值

# 轉為xgboost矩陣格式
trainData <- xgb.DMatrix(data = trainData, label = trainLabel)
testData <- xgb.DMatrix(data = testData, label = testLabel)

# 建立參數組合表
paramTable <- expand.grid(eta = c(0.3),                # 學習比率
                          max_depth = c(3, 4, 5),      # 深度
                          subsample = c(0.9),          # 每次疊代建立樹時隨機抽樣本的比率
                          colsample_bytree = c(0.9))   # 每次疊代建立樹時隨機抽變數的比率

# 進行交叉驗證挑選最佳參數
cvOutput <- NULL
for(iy in c(1:nrow(paramTable))){
  
  # 建立參數組合表
  params <- list(booster = "gbtree", 
                 eta = paramTable$eta[iy],
                 max_depth = paramTable$max_depth[iy], 
                 subsample = paramTable$subsample[iy],
                 colsample_bytree = paramTable$colsample_bytree[iy],
                 objective = "reg:squarederror")       # 目標函數: 迴歸
  
  # 進行交叉驗證
  cvResult <- xgb.cv(params = params,
                     data = trainData,              # 訓練集資料
                     nrounds = 2000,                # 最大疊代數
                     nfold = 5,                     # 折數
                     early_stopping_rounds = 20,    # 提早停止疊代樹
                     verbose = F)                   # 是否要印出學習過程資訊
  
  # 儲存此參數組合交叉驗證的結果
  cvOutput <- cvOutput %>%
    bind_rows(tibble(paramsNum = iy,
                     bestIteration = cvResult$best_iteration,
                     bestCvMeanError = cvResult$evaluation_log$train_rmse_mean[bestIteration]))
  
  gc()
}

# 交叉驗證最佳參數
bestCvSite <- which(cvOutput$bestCvMeanError == min(cvOutput$bestCvMeanError))
bestCvMeanError <- cvOutput$bestCvMeanError[bestCvSite]
bestIteration <- cvOutput$bestIteration[bestCvSite]
bestParamsNum <- cvOutput$paramsNum[bestCvSite]

# 最佳參數組合
params <- list(booster = "gbtree", 
               eta = paramTable$eta[bestParamsNum], 
               max_depth = paramTable$max_depth[bestParamsNum], 
               subsample = paramTable$subsample[bestParamsNum], 
               colsample_bytree = paramTable$colsample_bytree[bestParamsNum], 
               objective = "reg:squarederror")

# xgboost模型訓練
xgbModel <- xgb.train(data = trainData,
                      params = params,
                      nrounds = bestIteration)

# 模型特徵重要度
importance_matrix <- xgb.importance(xgbModel, 
                                    feature_names = colnames(trainData))
xgb.plot.importance(importance_matrix, top_n = 3)
xgb.ggplot.importance(importance_matrix, top_n = 3)

# 模型訓練集預測
predictTrainResult <- predict(xgbModel, trainData)

# 計算訓練集準確度
xgbModelTrainMSE <- MSE(y_real = trainLabel, y_pred = predictTrainResult)

# 模型訓練集預測
predictTestResult <- predict(xgbModel, testData)

# 計算訓練集準確度
xgbModelTestMSE <- MSE(y_real = testLabel, y_pred = predictTestResult)


################# 多層感知器模型(MLP) #################
library(keras)
# 執行前務必先安裝好相關軟體(Anaconda)及環境設定
# 請參考R語言Keras及XGBoost套件CPU/GPU版本安裝教學.ppt

# 將資料的y轉成數值格式 
trainData <- originTrainData
testData <- originTestData

# 整理訓練及測試集資料
trainLabel <- trainData %>% pull(y)
trainData <- trainData %>% select(-y) %>% data.matrix()  # 此處將所有的類別資料轉為數值
testLabel <- testData %>% pull(y)
testData <- testData %>% select(-y) %>% data.matrix()    # 此處將所有的類別資料轉為數值

# 深度學習模型架構
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 16, activation = "relu", input_shape = ncol(trainData)) %>%   # 第一層的input_shape對應到資料特徵數
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "linear")   # 最後一層的輸出為線性

# 觀看模型架構
summary(model)

# 設定深度學習模型
model %>% compile(
  loss = "mse",                        # 損失函數設定
  optimizer = optimizer_adam(),        # 學習優化器設定
  metrics = "mse")                     # 評估學習成效指標

# 設定提早終止學習條件
callbacks <- list(callback_early_stopping(monitor = "val_loss", patience = 10))
# patience: 若monitor連續n回合都沒下降 則停止學習

# 執行深度學習模型
history <- model %>% fit(
  trainData,                   # 訓練集資料
  trainLabel,                  # 訓練集標籤
  epochs = 100,                # 最大疊代次數
  batch_size = 32,             # batch樣本數
  callbacks = callbacks,       # 加入提早終止學習條件
  validation_split = 0.3)      # 交叉驗證樣本比率(自訓練集切割)

# 模型訓練集預測
predictTrainResult <- predict(model, trainData)

# 計算訓練集準確度
mlpModelTrainMSE <- MSE(y_real = trainLabel, y_pred = predictTrainResult)

# 模型訓練集預測
predictTestResult <- predict(model, testData)

# 計算訓練集準確度
mlpModelTestMSE <- MSE(y_real = testLabel, y_pred = predictTestResult)


############### 各模型預測結果比較 ################
cat(paste0("==========各模型預測集準確率==========\n", 
           "決策樹模型(Decision Tree)準確率: ", round(prunedTreeModelTestMSE, 2), " \n",
           "隨機森林模型(Random Forest)準確率: ", round(rfModelTestMSE, 2), " \n",
           "極限梯度提升模型(XGBoost)準確率: ", round(xgbModelTestMSE, 2), " \n",
           "多層感知器(MLP)準確率: ", round(mlpModelTestMSE, 2), " \n"))





