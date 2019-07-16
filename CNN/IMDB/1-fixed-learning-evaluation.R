library(keras)
library(dplyr)
library(ggplot2)
library(purrr)



##### Hyperparameters #####
  max_features <- 5000
  maxlen <- 100
  
  embedding_dims <- 50 # (output dimension of embedding layer)
  
  epochs <- 5
  batch_size <- 32
 
  
##### Data Preparation #####

  imdb <- dataset_imdb(num_words = max_features)
  word_index <- dataset_imdb_word_index()
  
  xdata <- imdb$train$x %>% pad_sequences(maxlen = maxlen)
  ydata <- imdb$train$y %>% to_categorical(num_class = 2)
  
  tr <- sample(1:nrow(xdata), round(nrow(xdata)*0.5))
  val <- sample(setdiff(1:nrow(xdata), tr), round(nrow(xdata)*0.3))
  te <- setdiff(setdiff(1:nrow(xdata), tr), val)
  
  xtrain <- xdata[tr,] ; xval <- xdata[val,] ; xtest <- xdata[te,]
  ytrain <- ydata[tr,] ; yval <- ydata[val,] ; ytest <- ydata[te,]

  
##### Review ##### 
  word_index_df <- data.frame(
    word = names(word_index),
    idx = unlist(word_index, use.names = FALSE),
    stringsAsFactors = FALSE
  ) %>% 
    mutate(idx = idx + 3) %>%
    add_row(word = "<PAD>", idx = 0) %>%
    add_row(word = "<START>", idx = 1) %>%
    add_row(word = "<UNK>", idx = 2) %>%
    add_row(word = "<UNUSED>", idx = 3) %>%
    arrange(idx)

  decode_review <- function(text) {
    paste(map(text, function(num) word_index_df %>%
              filter(idx == num) %>%
              select(word) %>%
              pull()),
          collapse = ' ')
  }
  
# (ex)  
  decode_review(xtrain[1,])

  
##### Modeling #####
  k_clear_session()
  
  model <- keras_model_sequential() 
  model %>%
    layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
    layer_conv_1d(32, 5, activation = 'relu', padding = 'same') %>%
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(32, 5, activation = 'relu',  padding = 'same') %>%
    layer_max_pooling_1d(2) %>%
    layer_conv_1d(32, 5, activation = 'relu',  padding = 'same') %>%
    layer_max_pooling_1d(2) %>%
    layer_flatten() %>%
    layer_dense(2, activation = 'softmax')
  
  model %>% 
    compile(optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = 'accuracy')

  
##### Training #####
  
# saving point  
  modelpath <- 'IMDB/Model/mymodel.hdf5'
  saving <- callback_model_checkpoint(
    filepath = modelpath,
    monitor = 'val_loss', save_best_only = TRUE, period = 1
  )

  history <- model %>% fit(
    xtrain, ytrain,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(xval, yval),
    callbacks = list(saving)
  )
  
  
##### Evaluating #####
  bestmodel <- load_model_hdf5(modelpath) # load the saved (best) model
  
  bestmodel %>% evaluate(xtest, ytest)  
  bestmodel %>% evaluate(xtrain, ytrain)  
  bestmodel %>% evaluate(xval, yval)  



