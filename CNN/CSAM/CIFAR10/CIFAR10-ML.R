library(tidyverse)
library(ranger)  # Random Forest
library(xgboost) # XGBoost
library(FNN)     # Fast KNN


# Class Labels
classInfo <- data.frame(
  Index = 0:9,
  Label = c('airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')
)


# 1. Data Preparation -------------------------------------------------------------------------------
# Load CIFAR10 DataSet 
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


# Training/Validation Split 
set.seed(100)
val <- sample(nrow(Xtrain), nrow(Xtrain)*0.2)
Xval <- Xtrain[val,,,]
Xtrain <- Xtrain[-val,,,]
Yval <- Ytrain[val,]
Ytrain <- Ytrain[-val,]


# Flatten Features & add Mean, SD for each channel 
data.flatten <- function(Xdata) {
  n_feature <- prod(dim(Xdata)[2:3])
  Xr <- Xdata[,,,1] %>% array_reshape(c(nrow(Xdata), n_feature))
  Xg <- Xdata[,,,2] %>% array_reshape(c(nrow(Xdata), n_feature)) 
  Xb <- Xdata[,,,3] %>% array_reshape(c(nrow(Xdata), n_feature))
  colnames(Xr) <- str_c('R', 1:n_feature)
  colnames(Xg) <- str_c('G', 1:n_feature)
  colnames(Xb) <- str_c('B', 1:n_feature)
  dat <- cbind(
    cbind(Xr, Rmean = rowMeans(Xr), Rsd = apply(Xr, 1, sd)),
    cbind(Xg, Gmean = rowMeans(Xg), Gsd = apply(Xg, 1, sd)),
    cbind(Xb, Bmean = rowMeans(Xb), Bsd = apply(Xb, 1, sd))
  )
  cat('Flattened Shape:', dim(dat), '\n')
  return(dat)
}

Xtrain_f <- data.flatten(Xtrain)
Xval_f <- data.flatten(Xval)
Xtest_f <- data.flatten(Xtest)

dtrain <- data.frame(Y = factor(Ytrain, 0:9), Xtrain_f)
dval <- data.frame(Y = factor(Yval, 0:9), Xval_f)
dtest <- data.frame(Y = factor(Ytest, 0:9), Xtest_f)


# 2. Leaning other ML models ------------------------------------------------------------------------
# Random Forest -------------------------------------------------------------------------------------
# tune grid
rfGrid <- expand.grid(mtry = c(32, 128, 256), ntree = c(300, 500), 
                      node_size = c(5, 9), sample_size = c(0.5, 0.9),
                      OOB_error = 0)

# Cross-Validation
for(i in 1:nrow(rfGrid)) {
  # Train model
  model <- ranger(Y ~ ., data = dtrain, seed = 100, 
                  num.trees = 500,
                  mtry = rfGrid$mtry[i], 
                  min.node.size = rfGrid$node_size[i], 
                  sample.fraction = rfGrid$sample_size[i])
  # Add OOB error to grid
  rfGrid$OOB_error[i] <- model$prediction.error
  print(i)
}

rfGrid <- rfGrid %>% arrange(OOB_error)
rfGrid

# Retrain using optimal parameters
dtot <- rbind(dtrain, dval)
rf.Final <- ranger(Y ~ ., data = dtot,
                   num.trees = rfGrid$ntree[1],
                   mtry = rfGrid$mtry[1],
                   min.node.size = rfGrid$node_size[1],
                   sample.fraction = rfGrid$sample_size[1],
                   importance = 'impurity')

# Prediction for the test set  
rf.Pred <- predict(rf.Final, dtest)


# XGBoost ------------------------------------------------------------------------------------------
# XGB data format
DMtrain <- xgb.DMatrix(Xtrain_f, label = Ytrain)
DMval <- xgb.DMatrix(Xval_f, label = Yval)
DMtest <- xgb.DMatrix(Xtest_f, label = Ytest)

# Tune grid
xgbGrid <- expand.grid(eta = c(0.01, 0.1, 0.2), max_depth = c(5, 10), gamma = c(0, 1))
watchList <- list(train = DMtrain, val = DMval)

# Cross-Validation
res <- NULL
for (i in 1:nrow(xgbGrid)) {
  params <- list(booster = 'gbtree', objective = 'multi:softprob', eval_metric = 'merror',
                 eta = xgbGrid$eta[i], max_depth = xgbGrid$max_depth[i], gamma = xgbGrid$gamma[i])
  
  model <- xgb.train(DMtrain, num_class = n_class, params = params, nrounds = 500,
                     watchlist = watchList, print_every_n = 5)
  
  tmp <- tail(model$evaluation_log, 1)
  res <- rbind(res, tmp)
}

# Retrain using optimal parameters
params <- xgbGrid[which.min(res$val_merror),]
DMtot <- xgb.DMatrix(rbind(Xtrain_f, Xval_f), label = c(Ytrain, Yval))
xgb.Final <- xgb.train(DMtot, num_class = n_class, params = params, nrounds = 500)

# Prediction for the test set  
xgb.Pred <- predict(xgb.Final, DMtest)


# KNN ----------------------------------------------------------------------------------------------
kGrid <- expand.grid(k = 1:5, val_error = 0)

# Cross-Validation
for (k in 1:nrow(kGrid)) {
  pred <- knn(dtrain[,-1], dval[,-1], dtrain$Y, k = kGrid$k[k], algorithm = 'kd_tree')
  kGrid$val_error[k] <- mean(pred != dval$Y)
}

kGrid <- kGrid %>% arrange(val_error)
kGrid

# Prediction for the test set  using optimal k  
knn.Pred <- knn(dtot[,-1], dtest[,-1], dtot$Y, k = 1, algorithm = 'kd_tree')


# 3. Evaluation for all models (including CNN) -----------------------------------------------------
cnn.Final <- load_model_hdf5('Model/model-user-3-bn.hdf5')
cnn.Pred <- cnn.Final %>% predict_classes(Xtest)

result <- data.frame(CNN = cnn.Pred, RF = rf.Pred, XGB = xgb.Pred, KNN = knn.Pred)
error.summary <- apply(result, 2, function(pred) round(mean(dtest$Y == pred), 4)*100)
error.tb <- data.frame(model = names(error.summary), acc = error.summary, stringsAsFactors = F)
error.tb <- error.tb %>% mutate(error = round(100-acc, 2))

# Figure 10
ggplot(error.tb) +
  theme_light() + theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5)) +
  geom_col(aes(fct_reorder(model, error), error, fill = model), color = 'gray50') +
  geom_text(aes(fct_reorder(model, error), error-5, label = error), size = 5) +
  scale_fill_brewer(palette = 'Pastel1') + 
  labs(x = '', y = 'Error (%)', title = 'Test Error Rates for all ML models') + 
  scale_x_discrete(labels = c(KNN1 = '1-NN')) + 
  theme(legend.position = '', axis.text = element_text(size = 12),
        plot.title = element_text(size = 16))

