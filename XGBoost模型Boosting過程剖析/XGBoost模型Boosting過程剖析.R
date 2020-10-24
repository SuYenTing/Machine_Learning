## XGBoost模型Boosting過程剖析
## 程式碼撰寫: 中山財管研究助理 蘇彥庭
## 參考程式碼: https://github.com/dmlc/xgboost/blob/master/R-package/demo/boost_from_prediction.R
## 此程式碼附有簡報說明: XGBoost模型Boosting過程剖析說明.pptx
library(xgboost)
library(tidyverse)

###################################### CASE1. 預測數值 ######################################
rm(list = ls()); gc()
# 讀取數據(此為數值資料)
data("marketing", package = "datarium")
trainData <- marketing[c(1:100), !(colnames(marketing) %in% c("sales"))]
trainLabel <- marketing[c(1:100), c("sales")]
valdData <- marketing[c(101:150), !(colnames(marketing) %in% c("sales"))]
valdLabel <- marketing[c(101:150), c("sales")]
testData <- marketing[c(151:200), !(colnames(marketing) %in% c("sales"))]
testLabel <- marketing[c(151:200), c("sales")]
dtrain <- xgb.DMatrix(as.matrix(trainData), label = trainLabel)
dvald <- xgb.DMatrix(as.matrix(valdData), label = valdLabel)
dtest <- xgb.DMatrix(as.matrix(testData), label = testLabel)

# 建立watchlist
watchlist <- list(eval = dvald, train = dtrain)

# 建立參數
params <- list(max_depth = 2, eta = 0.1, objective = "reg:squarederror")

# 測試方法: XGBoost模型直接訓練10次疊代
bst <- xgb.train(params = params, data = dtrain, nrounds = 10, base_score = 0.3, watchlist = watchlist)

# 預測結果
directPredcitResult <- predict(bst, dtest)

# 測試方法: 分開訓練10次
saveModelList <- list()
iter <- 1
for(iter in c(1:10)){
  
  bst <- xgb.train(params = params, 
                   data = dtrain, 
                   nrounds = 1,                             # 每次迴圈只跑一次疊代
                   base_score = ifelse(iter == 1, 0.3, 0),  # 第一次疊代設置和原XGB模型相同 第二次以後設置固定為0
                   watchlist = watchlist)
  
  # 儲存模型
  saveModelList[[iter]] <- bst
  
  # Note: we need the margin value instead of transformed prediction in set_base_margin
  # do predict with output_margin=TRUE, will always give you margin values before logistic transformation
  ptrain <- predict(bst, dtrain, outputmargin = TRUE)
  pvald  <- predict(bst, dvald, outputmargin = TRUE)
  
  # set the base_margin property of dtrain and dvald
  # base margin is the base prediction we will boost from
  setinfo(dtrain, "base_margin", ptrain)
  setinfo(dvald, "base_margin", pvald)
}

# 預測結果
predictResult <- sapply(saveModelList, predict, dtest)
loopPredcitResult <- apply(predictResult, 1, "sum")

# 比較XGBoost模型 直接訓練10次疊代 與 分開訓練10次 結果
head(directPredcitResult, 20)
head(loopPredcitResult, 20)
# 小結: 直接跑10次與分開訓練10次之預測結果相同


###################################### CASE2. 預測類別 ######################################
rm(list = ls()); gc()
# 讀取數據(此為預測類別資料)
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data[c(1:3000), ], label = agaricus.train$label[c(1:3000)])
dvald <- xgb.DMatrix(agaricus.train$data[c(3001:nrow(agaricus.train$data)), ], 
                     label = agaricus.train$label[c(3001:nrow(agaricus.train$data))])
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

# 建立watchlist
watchlist <- list(eval = dvald, train = dtrain)

# 建立參數
params <- list(max_depth = 2, eta = 0.1, objective = "binary:logistic")

# 測試方法: XGBoost模型直接訓練10次疊代
bst <- xgb.train(params = params, data = dtrain, nrounds = 10, base_score = 0.9, watchlist = watchlist)

# 預測結果
directPredcitResult <- predict(bst, dtest)

# 測試方法: 分開訓練10次
saveModelList <- list()
iter <- 1
for(iter in c(1:10)){
  
  bst <- xgb.train(params = params, 
                   data = dtrain, 
                   nrounds = 1,                               # 每次迴圈只跑一次疊代
                   base_score = ifelse(iter == 1, 0.9, 0.5),  # 第一次疊代設置和原XGB模型相同 第二次以後設置固定為0.5
                   watchlist = watchlist)
  
  # 儲存模型
  saveModelList[[iter]] <- bst
  
  # Note: we need the margin value instead of transformed prediction in set_base_margin
  # do predict with output_margin=TRUE, will always give you margin values before logistic transformation
  ptrain <- predict(bst, dtrain, outputmargin = TRUE)
  pvald  <- predict(bst, dvald, outputmargin = TRUE)
  
  # set the base_margin property of dtrain and dvald
  # base margin is the base prediction we will boost from
  setinfo(dtrain, "base_margin", ptrain)
  setinfo(dvald, "base_margin", pvald)
}

# 預測結果
predictResult <- sapply(saveModelList, predict, dtest, outputmargin = TRUE)
loopPredcitResult <- apply(predictResult, 1, "sum")
LogisticTransformFunction <- function(x) 1/(1+exp(-x))
loopPredcitResult <- LogisticTransformFunction(loopPredcitResult)

# 比較XGBoost模型 直接訓練10次疊代 與 分開訓練10次 結果
head(directPredcitResult, 20)
head(loopPredcitResult, 20)
# 小結: 直接跑10次與分開訓練10次之預測結果相同


###################################### 補充: 相關議題探討######################################
# XGBoost官網參數說明頁面
# https://xgboost.readthedocs.io/en/latest/parameter.html
# base_score:
#     * The initial prediction score of all instances, global bias
#     * For sufficient number of iterations, changing this value will not have too much effect.

# R的XGBoost套件說明文件
# https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
# base_score: the initial prediction score of all instances, global bias. Default: 0.5
# base_margin: base margin is the base prediction Xgboost will boost from

# 此篇討論base_score和base_margin差異
# https://github.com/dmlc/xgboost/issues/5028
# Question: Differentiating between base_score and base_margin
# Answer: 
# I believe one initialises the score globally for all instances, the other allows setting it individually for
# each training instance. In the second case this would be useful for continuing training from the
# output of some other kind of model.

# 此篇討論base_score的設定
# https://github.com/dmlc/xgboost/issues/799
# Question: base_score default

