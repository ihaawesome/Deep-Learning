library(keras)
library(dplyr)
library(tidyr)
library(ggplot2)



##### Hyperparameters #####
  max_features <- 5000
  maxlen <- 100
  
  embedding_dims <- 50 # (output dimension of embedding layer)
  
  epochs <- 5
  batch_size <- 32
    
  convs <- c(1, 2, 3)
  filters <- c(16, 32, 64)
  windows <- c(3, 5, 7, 9)
  
  Parameters <- data.frame(
    Nconv = rep(convs, each = 12),
    Nfilter = rep(rep(filters, each = 4), 3),
    Window = rep(windows, 9)
  )
  
  
##### Data Preparation #####
  imdb <- dataset_imdb(num_words = max_features)
  
  xtrain <- imdb$train$x %>% pad_sequences(maxlen = maxlen)
  ytrain <- imdb$train$y


##### Functions #####
  
# Define a Model Structure  
  IMDBmodel <- function(nc, nf, nw) {
    
    k_clear_session()
    model <- keras_model_sequential()
    
    model %>% 
      layer_embedding(max_features, embedding_dims, input_length = maxlen)
      
    for (cc in 1:nc) {
      model %>%
        layer_conv_1d(nf, nw, activation = "relu") %>%
        layer_max_pooling_1d(2)
    }  
    model %>%
      layer_flatten() %>%
      layer_dense(1, activation = 'sigmoid')
    
    model %>% compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    )
    
    return(model)
  }  
  
# Auto-Training  
  Training <- function(model, datalist, modelname, verbose = 0, ...) {
    xtrain <- datalist$xtrain
    ytrain <- datalist$ytrain
    xval <- datalist$xval
    yval <- datalist$yval
    xtest <- datalist$xtest
    ytest <- datalist$ytest
    
    saving <- callback_model_checkpoint(
      filepath = paste0('IMDB/Model/', modelname, '.hdf5'),
      monitor = 'val_loss', save_best_only = TRUE, period = 1, verbose = verbose
    )
    
    history <- model %>%
      fit(
        xtrain, ytrain,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = list(saving),
        validation_data = list(xval, yval),
        verbose = verbose
      )
    
    bestmodel <- load_model_hdf5(paste0('IMDB/Model/', modelname, '.hdf5'))
    train_acc <- evaluate(bestmodel, xtrain, ytrain, verbose = verbose)$acc
    val_acc <- evaluate(bestmodel, xval, yval, verbose = verbose)$acc
    test_acc <- evaluate(bestmodel, xtest, ytest, verbose = verbose)$acc
    
    err <- 1 - c(train_acc, val_acc, test_acc)
    names(err) <- c('train_error', 'val_error', 'test_error')
    
    return(err)
  }

 
##### START SIMULATION #####  
  Nsim <- 50
  Ncase <- nrow(Parameters)

  res <- NULL
  
  for (i in 1:Nsim) {
    train_id <- sample(1:nrow(xtrain), size = as.integer(nrow(xtrain)*0.5))
    val_id <- sample(setdiff(1:nrow(xtrain), train_id), size = as.integer(nrow(xtrain)*0.3))
    test_id <- setdiff(1:nrow(xtrain), c(train_id, val_id))
    
    xtrain_s <- xtrain[train_id,]
    ytrain_s <- ytrain[train_id]
    xval_s <- xtrain[val_id,]
    yval_s <- ytrain[val_id]
    xtest_s <- xtrain[test_id,]
    ytest_s <- ytrain[test_id]
    datalist <- list(xtrain = xtrain_s, ytrain = ytrain_s,
                     xval = xval_s, yval = yval_s,
                     xtest = xtest_s, ytest = ytest_s)
    
    for (j in 1:Ncase) {
      print(paste(i, j))  
      model <- IMDBmodel(nc = Parameters[j,1], nf = Parameters[j,2], nw = Parameters[j,3])
      err <- Training(model, datalist, modelname = 'temp')
      res <- rbind(res, err)
    }
  }
  
  summ <- data.frame(
    Nconv = rep(Parameters$Nconv, 20),
    Nfilter = rep(Parameters$Nfilter, 20),
    Window = rep(Parameters$Window, 20),
    res
  )


##### Summary #####      
  summ %>% group_by(Nconv) %>%
    summarise(train = mean(train_error), val = mean(val_error), test = mean(test_error))
  summ %>% group_by(Nfilter) %>%
      summarise(train = mean(train_error), val = mean(val_error), test = mean(test_error))
  summ %>% group_by(Window) %>%
      summarise(train = mean(train_error), val = mean(val_error), test = mean(test_error))
  
  ggsumm <- gather(summ, set, error, -Nconv, -Nfilter, -Window) %>% 
    mutate(set = factor(set, c('train_error', 'val_error', 'test_error')),
           Nconv = factor(Nconv), Nfilter = factor(Nfilter), Window = factor(Window))
  
  g <- ggplot(ggsumm) + 
    theme_bw() + theme(legend.position = '') + 
    scale_fill_brewer(palette = 'Pastel1')
  
  g + geom_boxplot(aes(x = Nconv, y = error, group = Nconv, fill = Nconv)) + facet_grid(~set)
  g + geom_boxplot(aes(x = Nfilter, y = error, group = Nfilter, fill = Nfilter)) + facet_grid(~set)
  g + geom_boxplot(aes(x = Window, y = error, group = Window, fill = Window)) + facet_grid(~set)
  
  
