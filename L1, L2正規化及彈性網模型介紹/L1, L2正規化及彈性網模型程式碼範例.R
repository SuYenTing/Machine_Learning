## L1, L2正規化及彈性網模型程式碼範例
## 程式撰寫: 中山財管所 蘇彥庭 研究助理
rm(list=ls());gc()
library(glmnet)
options(scipen = 999)

# 取出mtcars資料集
mtcars <- mtcars

# 建立訓練集(train)與測試集(test)的解釋變數與被解釋變數資料集
set.seed(168)
trainIdx <- sample(c(1:nrow(mtcars)), nrow(mtcars)*0.8)
trainData <- mtcars[trainIdx, ]
testData <- mtcars[-trainIdx, ]
trainX = model.matrix(mpg ~ 0 + ., data = trainData)
trainY = as.matrix(trainData["mpg"])
testX = model.matrix(mpg ~ 0 + ., data = testData)
testY = as.matrix(testData["mpg"])


###### 驗證Lasso和Ridge迴歸在懲罰參數=0時係數是否等於線性迴歸模型 ######

# 簡單線性迴歸模型
lmModel <- lm(mpg ~ ., data = trainData)
lmCoef <- coef(lmModel)

# 以簡單線性迴歸模型預測測試(test)資料集
lmPredict <- predict(lmModel, testData)

# Ridge迴歸且懲罰參數設0 = 簡單線性迴歸
ridgeModel <- glmnet(trainX, trainY, family = "gaussian", alpha = 0, lambda = 0, thresh = 1e-16)
ridgeCoef <- coef.glmnet(ridgeModel, s = 0)

# Lasso迴歸且懲罰參數設0 = 簡單線性迴歸 
lassoModel <- glmnet(trainX, trainY, family = "gaussian", alpha = 1, lambda = 0, thresh = 1e-16)
lassoCoef <- coef.glmnet(ridgeModel, s = 0)

# 整理各模型解釋變數係數表 驗證解釋變數係數是否相等
coefTable <- data.frame(lm = lmCoef, 
                        ridge = ridgeCoef[1:length(ridgeCoef)], 
                        lasso = lassoCoef[1:length(lassoCoef)])


###### Lasso和Ridge迴歸調參方法 ######

# Ridge迴歸調懲罰參數
ridgeModel <- glmnet(trainX, trainY, family = "gaussian", alpha = 0)
set.seed(168)
ridgeCv <- cv.glmnet(trainX, trainY, alpha = 0, nfolds = 5)
ridgeBestLambda <- ridgeCv$lambda.min
ridgeCoef <- coef.glmnet(ridgeModel, s = ridgeBestLambda)

# 以Ridge迴歸預測測試(test)資料集
ridgePredict <- predict(ridgeModel, s = ridgeBestLambda, newx = testX)

# Lasso迴歸調懲罰參數
lassoModel <- glmnet(trainX, trainY, family = "gaussian", alpha = 1)
set.seed(168)
lassoCv <- cv.glmnet(trainX, trainY, alpha = 1, nfolds = 5)
lassoBestLambda <- lassoCv$lambda.min
lassoCoef <- coef.glmnet(lassoModel, s = lassoBestLambda)

# 以Lasso迴歸預測測試(test)資料集
lassoPredict <- predict(lassoModel, s = ridgeBestLambda , newx = testX)

# 各模型解釋變數係數表
coefTable <- data.frame(lm = lmCoef, 
                        ridge = ridgeCoef[1:length(ridgeCoef)], 
                        lasso = lassoCoef[1:length(lassoCoef)])

# 視覺化調參過程
par(mfrow=c(2,2))
plot(ridgeCv, main = "Ridge")
plot(lassoCv, main = "Lasso")
plot(ridgeModel, xvar = "lambda")
plot(lassoModel, xvar = "lambda")


###### 彈性網(Elastic Net)調參方法 ######

# 迴圈Alpha值紀錄交叉驗證誤差值
tuneAlphaResult <- NULL
for(alpha in seq(0, 1, 0.1)){
  
  elasticNetModel <- glmnet(trainX, trainY, family = "gaussian", alpha = alpha)
  set.seed(168)
  elasticNetCv <- cv.glmnet(trainX, trainY, alpha = alpha, nfolds = 5)
  tuneAlphaResult <- rbind(tuneAlphaResult, 
                           data.frame(alpha = alpha, 
                                      lamda = elasticNetCv$lambda.min, 
                                      cvm= min(elasticNetCv$cvm)))
}

# 挑選最佳參數值
minCvmSite <- which(tuneAlphaResult$cvm==min(tuneAlphaResult$cvm))  # 最小交叉驗證平均誤差值的位置
bestAlpha <- tuneAlphaResult$alpha[minCvmSite]                      # 最佳alpha值
bestLambda <- tuneAlphaResult$lamda[minCvmSite]                     # 最佳lambda值

# 建立彈性網模型
elasticNetModel <- glmnet(trainX, trainY, family = "gaussian", alpha = bestAlpha)
elasticNetCoef <- coef.glmnet(elasticNetModel, s = bestLambda)

# 以彈性網模型預測測試(test)資料集
elasticNetPredict <- predict(elasticNetModel, s = bestLambda , newx = testX)

# 比較linear, Ridge, Lasso及Elastic Net預測結果和真實結果誤差
predictTable <- data.frame(testY, lmPredict, ridgePredict, lassoPredict, elasticNetPredict)
colnames(predictTable) <- c("test", "lm", "ridge", "lasso", "elasticNet")

# 比較各模型預測的MSE
mseComapre <- apply(predictTable, 2, 
                    function(x){ sum((x - predictTable$test)^2) / length(predictTable$test) })
mseComapre


