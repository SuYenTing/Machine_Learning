# 瞏厩策家絛ㄒ_も糶计侩醚
# 更keras甅ン
library(keras)

# 弄计沮栋
mnist <- dataset_mnist()
trainData <- mnist$train$x
trainClass <- mnist$train$y
testData <- mnist$test$x
testClass <- mnist$test$y

# 俱瞶癡絤栋籔代刚栋戈才家Α璶―
trainData <- array_reshape(trainData, c(nrow(trainData), 784))
testData <- array_reshape(testData, c(nrow(testData), 784))

# 疭紉夹非て
trainData <- trainData / 255
testData <- testData / 255

# 箇代ヘ夹锣店览跑计才家Α璶―
trainClass <- to_categorical(trainClass, 10)

# 瞏厩策家琜篶
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

# 芠家琜篶
summary(model)

# 砞﹚瞏厩策家
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# 磅︽瞏厩策家
history <- model %>% fit(
  trainData, trainClass, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# 芠癡絤Θ狦
plot(history)

# 箇代家计だ摸
predictClass <- model %>% predict_classes(testData)

# 龟悔/箇代だ摸痻皚
confusionMatrix <- table(predictClass,testClass)

# 璸衡箇代非絋瞯
accurRate <- sum(diag(confusionMatrix))/length(testClass)



