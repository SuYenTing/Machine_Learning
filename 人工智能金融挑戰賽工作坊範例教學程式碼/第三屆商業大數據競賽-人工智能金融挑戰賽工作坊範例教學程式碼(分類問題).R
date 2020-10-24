# 20200926_第三屆商業大數據競賽-人工智能金融挑戰賽工作坊範例教學程式碼(分類問題)
# 清空資料及記憶體
rm(list = ls());gc()

# 載入套件
library(data.table)       # 讀檔套件
library(tidyverse)        # 整理資料套件
library(ggpubr)           # 用來編排圖片的套件
options(scipen = 999)     # 數字不要以科學記號呈現

# 設定檔案載入位置 請自行設定檔案路徑
filePath <- "C:/Users/adm/Desktop/人工智能金融挑戰賽工作坊範例教學程式碼"
setwd(filePath)


################# 載入檔案 #################
# 載入訓練集資料
trainData <- fread("bank-full.csv", header = T, stringsAsFactors = F, data.table = F) %>%
  as_tibble()

# 載入測試集資料
testData <- fread("bank.csv", header = T, stringsAsFactors = F, data.table = F) %>%
  as_tibble()


################# 探索式資料分析(Exploratory Data Analysis; EDA) #################
#----------------------- 資料敘述統計學分析 -----------------------
# 查看資料欄位
str(trainData)

# 查看資料是否有缺值(NA)
sum(is.na(trainData))          # 比對資料是否有NA值
sum(trainData == "unknown")    # 比對資料是否有unkonwn值

# R內建的敘述統計函式
summary(trainData)

# 使用自己的敘述統計函式
# 敘述統計的函式
StatisticAnalysis <- function(featureName){
  
  featureData <- checkData %>% select(featureName) %>% pull(featureName)
  
  table <- tibble(featureName = featureName,
                  mode = mode(featureData),
                  length = sum(is.na(featureData) == F),          # 變數有資料的筆數
                  kind = length(unique(featureData)),             # 變數的種類 
                  na_count = sum(is.na(featureData)),             # NA值的數量
                  inf_count = sum(is.finite(featureData)),        # Inf值的數量
                  nan_count = sum(is.nan(featureData)),           # NaN值的數量
                  Min = min(as.numeric(featureData)),
                  Q1 = quantile(as.numeric(featureData), probs = 0.25, na.rm = T),
                  Median = median(as.numeric(featureData)),
                  Mean = mean(as.numeric(featureData)),
                  Q3 = quantile(as.numeric(featureData), probs = 0.75, na.rm = T),
                  Max = max(as.numeric(featureData)),
                  sd = sd(as.numeric(featureData)))
  
  return(table)
}

# 查看訓練集各變數的狀況
checkData <- trainData                                             # 將訓練集資料代為目前要查看的資料集
trainFeatureNames <- colnames(checkData)                           # 要查看的變數名稱
statTableTrain <- map_dfr(trainFeatureNames, StatisticAnalysis)    # 將訓練集資料做成敘述統計表
print(statTableTrain)                                              # 查看敘述統計表

# 查看測試集各變數的狀況
checkData <- testData
testFeatureNames <- colnames(checkData)
statTableTest <- map_dfr(testFeatureNames, StatisticAnalysis)
print(statTableTest)

# 查看各數據的分配狀況
# 連續型變數
continuousFeature <- statTableTrain$featureName[which(statTableTrain$mode == "numeric")]

# 繪製分配圖
i <- 1
for (i in 1:length(continuousFeature)) {
  
  factorNames <- continuousFeature[i]
  
  # 繪製數值分配
  pic1 <- trainData %>%
    select(factor = factorNames) %>%
    ggplot(aes(factor))+
    geom_density()+
    labs(x = "", y = "density",title = factorNames)
  
  # 依據預測目標分組繪製各組之數值分配
  pic2 <- trainData %>%
    select(factor = factorNames, y) %>%
    ggplot(aes(factor, fill = factor(y))) +   # 此處加入fill參數
    geom_density(alpha = 0.3) +               # alpha參數控制透明度
    labs(x = "", y = "density", fill = "y", title = factorNames)
  
  # 合併圖形
  g <- ggarrange(pic1, pic2, ncol = 2, nrow = 1)
  # print(g)
  
  # # 儲存圖形
  # ggsave(filename = paste0(factorNames,".png"), width = 9, height = 6)
}

# 離散型變數
discreteFeature <- statTableTrain$featureName[which(statTableTrain$mode == "character")]
discreteFeature <- discreteFeature[which(discreteFeature != "y")]  # 排除預測目標

# 繪製分配圖
ix <- 1
for (ix in 1:length(discreteFeature)) {
  
  factorNames <- discreteFeature[ix]
  
  pic1 <- trainData %>%
    select(factor = factorNames) %>%
    group_by(factor) %>%
    summarise(ratio = n()/nrow(trainData)) %>%
    ggplot(aes(x = factor, y = ratio, fill = factor))+
    geom_bar(stat = "identity")+
    labs(x = "", y = "ratio", title = factorNames, fill = factorNames)+
    theme(axis.text.x = element_text(angle = 90))
  
  pic2 <- trainData %>%
    select(factor = factorNames, y) %>%
    group_by(factor, y) %>%
    summarise(count = n()) %>%
    group_by(factor) %>%
    mutate(ratio = count/sum(count)) %>%
    ggplot(aes(x = factor, y = ratio, fill = factor(y)))+
    geom_bar(stat = "identity")+
    labs(x = "",y = "ratio",fill = "y",title = factorNames)+
    theme(axis.text.x = element_text(angle = 90))
  
  # 合併圖形
  g <- ggarrange(pic1, pic2, ncol = 2, nrow = 1)
  # print(g)
  
  # # 儲存圖形
  # ggsave(filename = paste0(factorNames,".png"), width = 9, height = 6)
}

# 取出訓練集中為數值型的變數並查看其相關係數的大小以及與 y 之間的相關係數
corData <- trainData %>% 
  select(statTableTrain$featureName[statTableTrain$mode == 'numeric'], y) %>%
  mutate(y = ifelse(y == "yes", 1, 0))

# 查看y的樣本分配比率
trainData %>%
  mutate(y = ifelse(y == "no", 0, 1)) %>%
  group_by(y) %>%
  summarise(count = n()) %>%
  mutate(ratio = round(count/sum(count),4)*100) %>%
  ggplot(aes(x = as.character(y), y = count, fill = as.character(y)))+
  geom_bar(stat = "identity")+
  labs(x = "是否會申辦定期存款", y = "樣本數量", fill = "類別(1:是 0:否)")+
  geom_text(aes(label = count, vjust = -0.5))+
  geom_text(aes(label = paste0(ratio, "%"), vjust = 1.5))


#----------------------- 變數間相關係數分析 -----------------------
# 載入相關係數圖的套件
library(corrplot)

# 計算相關係數
corPlot <- cor(corData, use = "pairwise.complete.obs")
# 繪製相關係數圖
corrplot(corPlot, 
         method = "color", 
         type = "lower",
         order = "hclust", 
         addCoef.col = "black",
         tl.col = "black", 
         tl.srt = 20,
         diag = FALSE)    
# 由相關係數可以看到對y來說只有duration的相關係數明顯

# 選取pdays中非為-1的樣本(表示客戶在過去的行銷活動中有聯絡過)繪製相關係數圖
corPlot <- cor(corData[corData$pdays != -1, ], use = "pairwise.complete.obs")
# 繪製相關係數圖
corrplot(corPlot, 
         method = "color", 
         type = "lower",
         order = "hclust", 
         addCoef.col = "black",
         tl.col = "black", 
         tl.srt = 20,
         diag = FALSE) 

# 選取pdays中為-1的樣本(表示客戶在過去的行銷活動中從未聯絡過)繪製相關係數圖
corPlot <- cor(corData[corData$pdays == -1,] %>% 
                 select(-pdays, -previous), use = "pairwise.complete.obs")
# 繪製相關係數圖
corrplot(corPlot, 
         method = "color", 
         type = "lower",
         order = "hclust", 
         addCoef.col = "black",
         tl.col = "black", 
         tl.srt = 20,
         diag = FALSE)


################# 資料前置處理 #################
# 將連續型變數做標準化
standardizeCol <- statTableTrain %>% 
  mutate(row = row_number()) %>% 
  filter(mode == "numeric") %>% 
  pull(row)

# 進行特徵標準化
for(iy in c(standardizeCol)){
  
  # 取出訓練期及測試期特徵資料
  trainFeature <- trainData %>% pull(iy)
  testFeature <- testData %>% pull(iy)
  
  # 計算特徵平均值及標準差
  trainFeatureMean <- mean(trainFeature)  # 計算特徵平均值
  trainFeatureSd <- sd(trainFeature)      # 計算特徵標準差
  
  # Z分數
  trainFeature <- (trainFeature-trainFeatureMean)/trainFeatureSd
  testFeature <- (testFeature-trainFeatureMean)/trainFeatureSd
  
  # 刪除離群值(大於正負3倍標準差)
  trainFeature[which(trainFeature > 3)] <- 3
  trainFeature[which(trainFeature < (-3))] <- (-3)
  testFeature[which(testFeature > 3)] <- 3
  testFeature[which(testFeature < (-3))] <- (-3)
  
  # 將整理好特徵取代回原資料
  trainData[, iy] <- trainFeature
  testData[, iy] <- testFeature
}

# 將離散型變數做factor
discreteCol <- statTableTrain %>% 
  mutate(row = row_number()) %>% 
  filter(mode == "character") %>%
  filter(featureName != "y") %>%     # 由於各模型的y格式要求不一樣 故先不做調整
  pull(row)

trainData[, discreteCol] <- lapply(trainData[, discreteCol], factor)
testData[, discreteCol] <- lapply(testData[, discreteCol], factor)

# 將訓練及測試集資料另外儲存 因為不同的模型所需要的Y和X格式會不相同
str(trainData)
str(testData)
originTrainData <- trainData
originTestData <- testData


################# 決策樹模型(Decision Tree) #################
# 匯入決策樹套件
library(rpart)            # 決策樹套件
library(partykit)         # 決策樹繪圖套件

# 將資料的y轉成factor格式
trainData <- originTrainData %>% mutate(y = factor(y, level = c("yes", "no")))
testData <- originTestData %>% mutate(y = factor(y, level = c("yes", "no")))

# 決策樹參數設定
rpartConfigure <- rpart.control(minsplit = 100,
                                minbucket = round(100/3),
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
treeModel <- rpart(formula = y ~ ., data = trainData,
                   method = "class",
                   control = rpartConfigure)

# 模型訓練集預測
predictTrainResult <- predict(treeModel, trainData, type = c("class"))
# predict函數的type引數可設為prob返回機率值

# 計算訓練集準確度
treeModelTrainAcc <- sum(predictTrainResult == trainData$y)/length(predictTrainResult)

# 模型測試集預測
predictTestResult <- predict(treeModel, testData, type = c("class"))

# 計算測試集準確度
treeModelTestAcc <- sum(predictTestResult == testData$y)/length(predictTestResult)

# 視覺化交叉驗證後結果
plotcp(treeModel)

# 繪製決策樹
plot(as.party(treeModel))
# rpart.plot(cartModel)

# 進行剪枝(選取交叉驗證錯誤率(xerror)最小的cp)
bestCp <- treeModel$cptable[which.min(treeModel$cptable[, "xerror"]), "CP"]
prunedTreeModel <- prune(treeModel, cp = bestCp)

# 模型訓練集預測
predictTrainResult <- predict(prunedTreeModel, trainData, type = c("class"))

# 計算訓練集準確度
prunedTreeModelTrainAcc <- sum(predictTrainResult == trainData$y)/length(predictTrainResult)

# 模型測試集預測
predictTestResult <- predict(prunedTreeModel, testData, type = c("class"))

# 計算測試集準確度
prunedTreeModelTestAcc <- sum(predictTestResult == testData$y)/length(predictTestResult)

# 視覺化剪枝結果
plotcp(prunedTreeModel)

# 繪製剪枝後決策樹
plot(as.party(prunedTreeModel))
# rpart.plot(prunedTreeModel)

# 觀看剪枝後決策樹重要變數
attr(prunedTreeModel$variable.importance, "names")

# 準確度比較
treeModelTrainAcc           # 剪枝前訓練期預測準確率
treeModelTestAcc            # 剪枝前測試期預測準確率
prunedTreeModelTrainAcc     # 剪枝後訓練期預測準確率
prunedTreeModelTestAcc      # 剪枝後測試期預測準確率

# 建立混淆矩陣
confuseMatrix <- table(predictTestResult, testData$y, dnn = c("預測", "實際"))

TP <- confuseMatrix[1, 1]   # 預測有申購短期存款而實際有申購短期存款(TP, True Positive)
FP <- confuseMatrix[1, 2]   # 預期有申購短期存款而實際沒有申購短期存款(FP, False Positive)
FN <- confuseMatrix[2, 1]   # 預測沒有申購短期存款而實際有申購短期存款(FN, False Negative)
TN <- confuseMatrix[2, 2]   # 預測沒有申購短期存款而實際沒有申購短期存款(TN, True Negative)
precision <- TP/(TP+FP)
recall <- TP/(TP+FN)
accuracy <- (TP+TN)/(TP+FP+FN+TN)
f1Score <- 2*precision*recall/(precision + recall)


################# 隨機森林模型(Random Forest) #################
# 載入套件
library(randomForest)

# 將資料的y轉成factor格式 
trainData <- originTrainData %>% mutate(y = factor(y, level = c("yes", "no")))
testData <- originTestData %>% mutate(y = factor(y, level = c("yes", "no")))

# 隨機森林模型
rfModel <- randomForest(y~., data = trainData, importance = T, ntree = 50)

# 查看隨機森林結果
rfModel

# 繪製隨機森林樹模型
plot(rfModel, lwd = 2)
legend("topright", colnames(rfModel$err.rate), lty = c(1:3), lwd = 2, col = c(1:3))
# 黑色線為Out of Bag錯誤率
# 紅色線為實際類別為Yes的Out of Bag(OOB)錯誤率
# 綠色線為實際類別為No的Out of Bag(OOB)錯誤率

# 觀看隨機森林的重要特徵
importanceTable <- importance(rfModel) %>%
  as_tibble() %>%
  mutate(feature = rownames(importance(rfModel))) %>%
  arrange(-MeanDecreaseGini) %>%
  select(feature, MeanDecreaseGini, everything())

# 繪製重要特徵圖形
importanceTable %>%
  ggplot(aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini))+
  geom_col()+
  coord_flip()

# 模型訓練集預測
predictTrainResult <- predict(rfModel, trainData) 

# 計算訓練集準確度
rfModelTrainAcc <- sum(predictTrainResult == trainData$y)/length(predictTrainResult)

# 模型測試集預測
predictTestResult <- predict(rfModel, testData) 

# 計算測試集準確度
rfModelTestAcc <- sum(predictTestResult == testData$y)/length(predictTestResult)

# 建立混淆矩陣
confuseMatrix <- table(predictTestResult, testData$y, dnn = c("預測", "實際"))

TP <- confuseMatrix[1, 1]   # 預測有申購短期存款而實際有申購短期存款(TP, True Positive)
FP <- confuseMatrix[1, 2]   # 預期有申購短期存款而實際沒有申購短期存款(FP, False Positive)
FN <- confuseMatrix[2, 1]   # 預測沒有申購短期存款而實際有申購短期存款(FN, False Negative)
TN <- confuseMatrix[2, 2]   # 預測沒有申購短期存款而實際沒有申購短期存款(TN, True Negative)
precision <- TP/(TP+FP)
recall <- TP/(TP+FN)
accuracy <- (TP+TN)/(TP+FP+FN+TN)
f1Score <- 2*precision*recall/(precision + recall)


################# 極限梯度提升模型(XGBoost) #################
library(xgboost)

# 將資料的y轉成數值格式 
trainData <- originTrainData %>% mutate(y = ifelse(y == "yes", 1, 0))
testData <- originTestData %>% mutate(y = ifelse(y == "yes", 1, 0))

# 整理訓練及測試集資料
trainLabel <- trainData %>% pull(y)
trainData <- sapply(trainData %>% select(-y), as.numeric) %>% data.matrix()  # 此處將所有的類別資料轉為數值
testLabel <- testData %>% pull(y)
testData <- sapply(testData %>% select(-y), as.numeric) %>% data.matrix()    # 此處將所有的類別資料轉為數值

# 轉為xgboost矩陣格式
trainData <- xgb.DMatrix(data = trainData, label = trainLabel)
testData <- xgb.DMatrix(data = testData, label = testLabel)

# 建立參數組合表
paramTable <- expand.grid(eta = c(0.3), 
                          max_depth = c(3, 4, 5), 
                          subsample = c(0.9), 
                          colsample_bytree = c(0.9))

# 進行交叉驗證挑選最佳參數
cvOutput <- NULL
for(iy in c(1:nrow(paramTable))){
  
  params <- list(booster = "gbtree", 
                 eta = paramTable$eta[iy],
                 max_depth = paramTable$max_depth[iy], 
                 subsample = paramTable$subsample[iy],
                 colsample_bytree = paramTable$colsample_bytree[iy],
                 objective = "binary:logistic")
  
  cvResult <- xgb.cv(params = params, 
                     data = trainData,
                     nrounds = 2000,
                     nfold = 5, 
                     early_stopping_rounds = 20,
                     verbose = F)
  
  cvOutput <- cvOutput %>%
    bind_rows(tibble(paramsNum = iy,
                     bestIteration = cvResult$best_iteration,
                     bestCvMeanError = cvResult$evaluation_log$train_error_mean[bestIteration]))
  
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
               objective = "binary:logistic")

# xgboost模型訓練
xgbModel <- xgb.train(data = trainData,
                      params = params,
                      nrounds = bestIteration)

# 模型特徵重要度
importance_matrix <- xgb.importance(xgbModel, feature_names = colnames(trainData))
xgb.plot.importance(importance_matrix, top_n = 5)
xgb.ggplot.importance(importance_matrix, top_n = 5)

# 模型訓練集預測
predictTrainResult <- predict(xgbModel, trainData) 
predictTrainResult <- as.numeric(predictTrainResult > 0.5)

# 計算訓練集準確度
xgbModelTrainAcc <- sum(predictTrainResult == trainLabel)/length(predictTrainResult)

# 模型測試集預測
predictTestResult <- predict(xgbModel, testData) 
predictTestResult <- as.numeric(predictTestResult > 0.5)

# 計算測試集準確度
xgbModelTestAcc <- sum(predictTestResult == testLabel)/length(predictTestResult)

# 建立混淆矩陣
confuseMatrix <- table(predictTestResult, testLabel, dnn = c("預測", "實際"))

TP <- confuseMatrix[1, 1]   # 預測有申購短期存款而實際有申購短期存款(TP, True Positive)
FP <- confuseMatrix[1, 2]   # 預期有申購短期存款而實際沒有申購短期存款(FP, False Positive)
FN <- confuseMatrix[2, 1]   # 預測沒有申購短期存款而實際有申購短期存款(FN, False Negative)
TN <- confuseMatrix[2, 2]   # 預測沒有申購短期存款而實際沒有申購短期存款(TN, True Negative)
precision <- TP/(TP+FP)
recall <- TP/(TP+FN)
accuracy <- (TP+TN)/(TP+FP+FN+TN)
f1Score <- 2*precision*recall/(precision + recall)


################# 多層感知器模型(MLP) #################
library(keras)
# 執行前務必先安裝好相關軟體(Anaconda)及環境設定
# 請參考R語言Keras及XGBoost套件CPU/GPU版本安裝教學.ppt

# 將資料的y轉成數值格式 
trainData <- originTrainData %>% mutate(y = ifelse(y == "yes", 1, 0))
testData <- originTestData %>% mutate(y = ifelse(y == "yes", 1, 0))

# 整理訓練及測試集資料
trainLabel <- trainData %>% pull(y)
trainData <- sapply(trainData %>% select(-y), as.numeric) %>% data.matrix()  # 此處將所有的類別資料轉為數值
testLabel <- testData %>% pull(y)
testData <- sapply(testData %>% select(-y), as.numeric) %>% data.matrix()    # 此處將所有的類別資料轉為數值

# 預測目標轉為虛擬變數以符合模型格式要求
trainLabelEncode <- to_categorical(trainLabel, num_classes = 2)

# 深度學習模型架構
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 32, activation = "relu", input_shape = ncol(trainData)) %>%   # 第一層的input_shape對應到資料特徵數
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = "softmax")   # 最後一層的輸出需對應到分類數

# 觀看模型架構
summary(model)

# 設定深度學習模型
model %>% compile(
  loss = "categorical_crossentropy",   # 損失函數設定
  optimizer = optimizer_adam(),        # 學習優化器設定
  metrics = c("accuracy")              # 評估學習成效指標
)

# 設定提早終止學習條件
callbacks <- list(callback_early_stopping(monitor = "val_loss", patience = 10))
# patience: 若monitor連續n回合都沒下降 則停止學習

# 執行深度學習模型
history <- model %>% fit(
  trainData,                   # 訓練集資料
  trainLabelEncode,            # 訓練集標籤
  epochs = 50,                 # 最大疊代次數
  batch_size = 128,            # batch樣本數
  callbacks = callbacks,       # 加入提早終止學習條件
  validation_split = 0.2)      # 交叉驗證樣本比率(自訓練集切割)

# 模型訓練集預測
predictTrainResult <- model %>% predict_classes(trainData)

# 計算訓練集準確度
mlpModelTrainAcc <- sum(predictTrainResult == trainLabel)/length(predictTrainResult)

# 模型測試集預測
predictTestResult <- model %>% predict_classes(testData)

# 計算測試集準確度
mlpModelTestAcc <- sum(predictTestResult == testLabel)/length(predictTestResult)

# 建立混淆矩陣
confuseMatrix <- table(predictTestResult, testLabel, dnn = c("預測", "實際"))

TP <- confuseMatrix[1, 1]   # 預測有申購短期存款而實際有申購短期存款(TP, True Positive)
FP <- confuseMatrix[1, 2]   # 預期有申購短期存款而實際沒有申購短期存款(FP, False Positive)
FN <- confuseMatrix[2, 1]   # 預測沒有申購短期存款而實際有申購短期存款(FN, False Negative)
TN <- confuseMatrix[2, 2]   # 預測沒有申購短期存款而實際沒有申購短期存款(TN, True Negative)
precision <- TP/(TP+FP)
recall <- TP/(TP+FN)
accuracy <- (TP+TN)/(TP+FP+FN+TN)
f1Score <- 2*precision*recall/(precision + recall)


############### 重要變數視覺化分析 ################
# 沖積圖
library(ggalluvial)

# 讀取訓練資料
trainData <- originTrainData

# 數值標籤化
CutGroup <- function(x) cut(x, 
                            c(min(x), quantile(x, 0.25), quantile(x, 0.5), quantile(x, 0.75), max(x)),
                            c("Q1", "Q2", "Q3", "Q4"), 
                            include.lowest = T)

# 整理資料
plotAlluvial <- trainData %>% 
  select(y, duration, balance, age) %>%
  mutate_at(.vars = c("duration", "balance", "age"), .funs = CutGroup) %>%
  group_by(y, duration, balance, age) %>%
  summarise(freq = n())

# 繪製圖形
plotAlluvial %>%
  ggplot(aes(y = freq, axis1 = y, axis2 = duration, axis3 = balance, axis4 = age)) +
  geom_alluvium(aes(fill = y)) +
  geom_stratum() +
  geom_label(stat = "stratum", infer.label = TRUE) +
  scale_x_discrete(limits = c("y", "duration", "balance","age"), expand = c(.05, .05)) +
  ggtitle("Bank Deposit Prediction")


############### 各模型預測結果比較 ################
cat(paste0("==========各模型預測集準確率==========\n", 
           "決策樹模型(Decision Tree)準確率: ", round(prunedTreeModelTestAcc*100, 2), "% \n",
           "隨機森林模型(Random Forest)準確率: ", round(rfModelTestAcc*100, 2), "% \n",
           "極限梯度提升模型(XGBoost)準確率: ", round(xgbModelTestAcc*100, 2), "% \n",
           "多層感知器(MLP)準確率: ", round(mlpModelTestAcc*100, 2), "% \n"))





