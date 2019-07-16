library(keras)
library(tidyverse)


# 10 Class Labels
classInfo <- data.frame(
  Index = 0:9,
  Label = c('airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')
)


# 1. Data Preparation -----------------------------------------------------------------------------------
# Load CIFAR10 DataSet from Keras 
cifar <- dataset_cifar10()
c(Xtrain, Ytrain) %<-% cifar$train
c(Xtest, Ytest) %<-% cifar$test
cat('Dim of Images in Training + Validation Set:', dim(Xtrain), '\n')
cat('Dim of Images in Test Set:', dim(Xtest), '\n')
cat('\n')
cat('Number of Labels per class in Training + Validation Set:', '\n')
print(table(class = Ytrain))
cat('\n')
cat('Number of Labels per class in Test Set:', '\n')
print(table(class = Ytest))


n_class <- 10
img_size <- 32
Xtrain <- Xtrain / 255
Xtest <- Xtest / 255
Ytrain <- Ytrain %>% to_categorical(n_class)
Ytest <- Ytest %>% to_categorical(n_class)


# Training/Validation Split 
set.seed(100)
val <- sample(nrow(Xtrain), nrow(Xtrain)*0.2)
Xval <- Xtrain[val,,,]
Yval <- Ytrain[val,]
Xtrain <- Xtrain[-val,,,]
Ytrain <- Ytrain[-val,]


# 2. Learning CNN models --------------------------------------------------------------------------------
# M1 ----------------------------------------------------------------------------------------------------
M1 <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = c(img_size, img_size, 3),
                16, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform',
                name = '1_conv') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '1_pool') %>%
  layer_conv_2d(32, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '2_conv') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '2_pool') %>%
  layer_conv_2d(64, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '3_conv') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '3_pool') %>%
  layer_flatten(name = 'flatten') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M1 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath1 <- 'Model/model-user-1.hdf5'
Msave1 <- callback_model_checkpoint(Mpath1, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 20)

epochs <- 100
batch_size <- 32

history1 <- M1 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size,
  callbacks = list(Msave1, stopping), verbose = 2
)


# M2 ----------------------------------------------------------------------------------------------------
M2 <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = c(img_size, img_size, 3),
                16, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '1_conv1') %>%
  layer_conv_2d(16, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '1_conv2') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '1_pool') %>%
  layer_conv_2d(32, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '2_conv1') %>%
  layer_conv_2d(32, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '2_conv2') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '2_pool') %>%
  layer_conv_2d(64, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '3_conv1') %>%
  layer_conv_2d(64, c(3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', 
                name = '3_conv2') %>%
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '3_pool') %>%
  layer_flatten(name = 'flatten') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M2 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath2 <- 'Model/model-user-2.hdf5'
Msave2 <- callback_model_checkpoint(Mpath2, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 20)

epochs <- 100
batch_size <- 32

history2 <- M2 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size,
  callbacks = list(Msave2, stopping), verbose = 2
)


# M3 ----------------------------------------------------------------------------------------------------
M3 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3),
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>% 
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>% 
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '2_pool') %>%
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>% 
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '3_pool') %>%
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M3 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath3 <- 'Model/model-user-2-bn.hdf5'
Msave3 <- callback_model_checkpoint(Mpath3, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 30)

epochs <- 100
batch_size <- 32

history3 <- M3 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size,
  callbacks = list(Msave3, stopping), verbose = 2
)


# M4 ----------------------------------------------------------------------------------------------------
M4 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3), 
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '2_pool') %>%	
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '3_pool') %>%	
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '1_dense') %>%
  layer_dropout(0.1, name = '1_dropout') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M4 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath4 <- 'Model/model-user-2-bn-fc-do-1.hdf5'
Msave4 <- callback_model_checkpoint(Mpath4, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 50)

epochs <- 300
batch_size <- 32

history4 <- M4 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size, 
  callbacks = list(Msave4, stopping), verbose = 2
)


# M5 ----------------------------------------------------------------------------------------------------
M5 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3), 
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '2_pool') %>%	
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>%
  layer_max_pooling_2d(c(2, 2), name = '3_pool') %>%	
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '1_dense') %>%
  layer_dropout(0.1, name = '1_dropout') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '2_dense') %>%
  layer_dropout(0.1, name = '2_dropout') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M5 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath5 <- 'Model/model-user-2-bn-fc-do-2.hdf5'
Msave5 <- callback_model_checkpoint(Mpath5, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 50)

epochs <- 300
batch_size <- 32

history5 <- M5 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size, 
  callbacks = list(Msave5, stopping), verbose = 2
)


# M6 ----------------------------------------------------------------------------------------------------
M6 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3),
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>% 
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>% 
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv3') %>%
  layer_batch_normalization(name = '1_bn3') %>%
  layer_activation_relu(name = '1_activation3') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>% 
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>% 
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv3') %>%
  layer_batch_normalization(name = '2_bn3') %>%
  layer_activation_relu(name = '2_activation3') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '2_pool') %>%
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>% 
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>% 
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv3') %>%
  layer_batch_normalization(name = '3_bn3') %>%
  layer_activation_relu(name = '3_activation3') %>% 
  layer_max_pooling_2d(c(2, 2), padding = 'same', name = '3_pool') %>%
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M6 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath6 <- 'Model/model-user-3-bn.hdf5'
Msave6 <- callback_model_checkpoint(Mpath6, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 20)

epochs <- 100
batch_size <- 32

history6 <- M6 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size,
  callbacks = list(Msave6, stopping), verbose = 2
)


# M7 ----------------------------------------------------------------------------------------------------
M7 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3), 
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '1_conv3') %>%
  layer_batch_normalization(name = '1_bn3') %>%
  layer_activation_relu(name = '1_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '2_conv3') %>%
  layer_batch_normalization(name = '2_bn3') %>%
  layer_activation_relu(name = '2_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '2_pool') %>%	
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_normal', name = '3_conv3') %>%
  layer_batch_normalization(name = '3_bn3') %>%
  layer_activation_relu(name = '3_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '3_pool') %>%	
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '1_dense') %>%
  layer_dropout(0.1, name = '1_dropout') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M7 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath7 <- 'Model/model-user-3-bn-fc-do-1.hdf5'
Msave7 <- callback_model_checkpoint(Mpath7, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 50)

epochs <- 300
batch_size <- 32

history7 <- M7 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size, 
  callbacks = list(Msave7, stopping), verbose = 2
)


# M8 ----------------------------------------------------------------------------------------------------
M8 <- keras_model_sequential() %>%
  
  layer_conv_2d(input_shape = c(img_size, img_size, 3), 
                16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv1') %>%
  layer_batch_normalization(name = '1_bn1') %>%
  layer_activation_relu(name = '1_activation1') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv2') %>%
  layer_batch_normalization(name = '1_bn2') %>%
  layer_activation_relu(name = '1_activation2') %>%
  layer_conv_2d(16, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '1_conv3') %>%
  layer_batch_normalization(name = '1_bn3') %>%
  layer_activation_relu(name = '1_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '1_pool') %>%
  
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv1') %>%
  layer_batch_normalization(name = '2_bn1') %>%
  layer_activation_relu(name = '2_activation1') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv2') %>%
  layer_batch_normalization(name = '2_bn2') %>%
  layer_activation_relu(name = '2_activation2') %>%
  layer_conv_2d(32, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '2_conv3') %>%
  layer_batch_normalization(name = '2_bn3') %>%
  layer_activation_relu(name = '2_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '2_pool') %>%	
  
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv1') %>%
  layer_batch_normalization(name = '3_bn1') %>%
  layer_activation_relu(name = '3_activation1') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv2') %>%
  layer_batch_normalization(name = '3_bn2') %>%
  layer_activation_relu(name = '3_activation2') %>%
  layer_conv_2d(64, c(3, 3), padding = 'same', kernel_initializer = 'he_uniform', name = '3_conv3') %>%
  layer_batch_normalization(name = '3_bn3') %>%
  layer_activation_relu(name = '3_activation3') %>%
  layer_max_pooling_2d(c(2, 2), name = '3_pool') %>%	
  
  layer_flatten(name = 'flatten') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '1_dense') %>%
  layer_dropout(0.1, name = '1_dropout') %>%
  layer_dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = '2_dense') %>%
  layer_dropout(0.1, name = '2_dropout') %>%
  layer_dense(n_class, activation = 'softmax', name = 'output')

M8 %>% compile(
  loss = 'categorical_crossentropy',
  metrics = 'accuracy',
  optimizer = optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
)

Mpath8 <- 'Model/model-user-3-bn-fc-do-2.hdf5'
Msave8 <- callback_model_checkpoint(Mpath8, monitor = 'val_acc', save_best_only = TRUE)
stopping <- callback_early_stopping(monitor = 'val_acc', patience = 50)

epochs <- 300
batch_size <- 32

history8 <- M8 %>% fit(
  Xtrain, Ytrain, validation_data = list(Xval, Yval),
  epochs = epochs, batch_size = batch_size, 
  callbacks = list(Msave8, stopping), verbose = 2
)


# 3. Evaluation of all CNN models -----------------------------------------------------------------------
# Load all saved models 
M1 <- load_model_hdf5('Model/model-user-1.hdf5')
M2 <- load_model_hdf5('Model/model-user-2.hdf5')
M3 <- load_model_hdf5('Model/model-user-2-bn.hdf5')
M4 <- load_model_hdf5('Model/model-user-2-bn-fc-do-1.hdf5')
M5 <- load_model_hdf5('Model/model-user-2-bn-fc-do-2.hdf5')
M6 <- load_model_hdf5('Model/model-user-3-bn.hdf5')
M7 <- load_model_hdf5('Model/model-user-3-bn-fc-do-1.hdf5')
M8 <- load_model_hdf5('Model/model-user-3-bn-fc-do-2.hdf5')


# Evaluation for the test set 
result1 <- M1 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result2 <- M2 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result3 <- M3 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result4 <- M4 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result5 <- M5 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result6 <- M6 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result7 <- M7 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()
result8 <- M8 %>% evaluate(Xtest, Ytest, verbose = 2) %>% as.data.frame()

result.summary <- rbind(result1, result2, result3, result4, result5, result6, result7, result8)
result.summary <- result.summary %>% 
  mutate(model = paste0('M', 1:8), error = round(1-acc, 4)*100)

# Figure 9
ggplot(result.summary) + 
  theme_test() + theme(plot.title = element_text(hjust = 0.5), legend.position = '') + 
  scale_fill_brewer(palette = 'Pastel1') +
  geom_col(aes(fct_reorder(model, error), error, fill = model), color = 'gray50') + 
  geom_text(aes(model, error-2, label = error)) +
  labs(title = 'Test Error Rates of all CNNs', x = '', y = 'Error (%)')



