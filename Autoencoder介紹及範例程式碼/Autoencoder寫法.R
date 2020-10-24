# Autoencoder模型
# 程式碼翻寫自Youtuber周莫煩 https://www.youtube.com/watch?v=OubNgB-Fa4M，原本為python的寫法改為R的寫法
# R的Autoencoder寫法可參考:https://keras.rstudio.com/articles/examples/variational_autoencoder.html
# 程式碼步驟與周莫煩類似，但寫法更複雜
rm(list=ls());gc()

library(keras)
library(dplyr)
library(ggplot2)

# 讀取數據集
mnist <- dataset_mnist() 

# 切分訓練及測試資料
x_train <- mnist$train$x/255-0.5
x_test <- mnist$test$x/255-0.5
x_train <- x_train %>% apply(1, as.numeric) %>% t()
x_test <- x_test %>% apply(1, as.numeric) %>% t()

set.seed(1377)

# 訓練資料共60,000筆，784個特徵(28*28像素)
dim(x_train)
dim(x_test)

# 壓縮目標特徵維度
encoding_dim <- 2

# 壓縮部分
input_img <- layer_input(shape=784)
encoded <- layer_dense(input_img, 128, activation = "relu")
encoded <- layer_dense(encoded, 64, activation = "relu")
encoded <- layer_dense(encoded, 10, activation = "relu")
encoder_output <- layer_dense(encoded, encoding_dim)

# 解壓部分
decoded <- layer_dense(encoder_output, 10, activation = "relu")
decoded <- layer_dense(decoded, 64, activation = "relu")
decoded <- layer_dense(decoded, 128, activation = "relu")
decoded <- layer_dense(decoded, 784, activation = "tanh")

# 壓縮及解壓模型
autoencoder <- keras_model(inputs=input_img, outputs=decoded)

# 純壓縮模型
encoder <- keras_model(inputs=input_img, outputs=encoder_output)

# compile
compile(object=autoencoder, optimizer="adam", loss="mse")

# 訓練
autoencoder %>% fit(
  x=x_train,
  y=x_train, 
  epochs = 20, 
  batch_size = 256
)

# 預測
x_train_encoded <- predict(encoder, x_train)

# 繪製圖形
x_train_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$train$y)) %>%
  ggplot(aes(x = "特徵1", y = "特徵2", colour = class)) + geom_point()

# 預測
x_test_encoded <- predict(encoder, x_test)

# 繪製圖形
x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

