library(keras)
library(tidyverse)


# 1. Data Preaparation ------------------------------------------------------------
# Input image / Output dimensions 
img_rows <- 28
img_cols <- 28
num_classes <- 10

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
Xtrain <- mnist$train$x
Ytrain <- mnist$train$y
Xtest <- mnist$test$x
Ytest <- mnist$test$y

# Redefine  dimension of train/test inputs
Xtrain <- array_reshape(Xtrain, c(nrow(Xtrain), img_rows, img_cols, 1))
Xtest <- array_reshape(Xtest, c(nrow(Xtest), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
Xtrain <- Xtrain / 255
Xtest <- Xtest / 255

cat('Xtrain_shape:', dim(Xtrain), '\n')
cat(nrow(Xtrain), 'train samples\n')
cat(nrow(Xtest), 'test samples\n')

# Convert class vectors to binary class matrices
Ytrain <- to_categorical(Ytrain, num_classes)
Ytest <- to_categorical(Ytest, num_classes)


# 2-1. Learning the example model (Keras) -------------------------------------------
# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Train model
epochs <- 12
batch_size <- 128

history <- model %>% fit(
  Xtrain, Ytrain, validation_split = 0.2,
  epochs = epochs, batch_size = batch_size,
  verbose = 0
)

# Save the last model
save_model_weights_hdf5(model, 'Model/model-example.hdf5')


# 2-2. Learning another model witout dropouts --------------------------------------
model_nodrop <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')

model_nodrop %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

epochs <- 12
batch_size <- 128

history_nodrop <- model_nodrop %>% fit(
  Xtrain, Ytrain, validation_split = 0.2,
  epochs = epochs, batch_size = batch_size,
  verbose = 2
)

save_model_hdf5(model, 'model-nodrop.hdf5')


# 3. Evaluation of both models ----------------------------------------------------
scores <- model %>% evaluate(Xtest, Ytest, verbose = 0)
scores_nodrop <- model_nodrop %>% evaluate(x_test, y_test, verbose = 0)

as.data.frame(scores)
as.data.frame(scores_nodrop)

