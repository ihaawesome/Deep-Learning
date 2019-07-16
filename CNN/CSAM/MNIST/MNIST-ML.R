library(tidyverse)
library(keras)
library(ranger)
library(xgboost)
library(FNN)


# 1. Data Preparation ------------------------------------------------------------------
# Load Dataset
mnist <- dataset_mnist()
c(Xtrain, Ytrain) %<-% mnist$train
c(Xtest, Ytest) %<-% mnist$test

img_rows <- 28
img_cols <- 28
Xtrain <- Xtrain / 255
Xtest <- Xtest / 255

data.flatten <- function(Xdata) {
  n_feature <- prod(dim(Xdata)[2:3])
  X <- Xdata %>% array_reshape(c(nrow(Xdata), n_feature))
  colnames(X) <- paste0('X', 1:n_feature)
  dat <- cbind(X, Xmean = rowMeans(X), Xsd = apply(X, 1, sd))
  cat('Flattened Shape:', dim(dat), '\n')
  return(dat)
}

# Flatten each set
Xtrain_f <- data.flatten(Xtrain)  
Xtest_f <- data.flatten(Xtest)  

# for other ML models
dtrain <- data.frame(Y = factor(Ytrain, 0:9), Xtrain_f)
dtest <- data.frame(Y = factor(Ytest, 0:9), Xtest_f)

# for CNN
Xtrain <- Xtrain %>% array_reshape(c(nrow(Xtrain), img_rows, img_cols, 1))
Xtest <- Xtest %>% array_reshape(c(nrow(Xtrain), img_rows, img_cols, 1))
Ytrain <- Ytrain %>% to_categorical(num_classes)
Ytest <- Ytest %>% to_categorical(num_classes)


# 2. Learning other ML models -----------------------------------------------------------
# Random Forest -------------------------------------------------------------------------
rfGrid <- expand.grid(mtry = c(32, 128, 256), ntree = c(300, 500), 
                      node_size = c(5, 9), sample_size = c(0.5, 0.9),
                      OOB_error = 0)

for(i in 1:nrow(rfGrid)) {
  # train model
  model <- ranger(Y ~ ., data = dtrain, seed = 100, 
                  num.trees = 500,
                  mtry = rfGrid$mtry[i], 
                  min.node.size = rfGrid$node_size[i], 
                  sample.fraction = rfGrid$sample_size[i])
  # add OOB error to grid
  rfGrid$OOB_error[i] <- model$prediction.error
  print(i)
}

rfGrid <- rfGrid %>% arrange(OOB_error)
rfGrid

rf.Final <- ranger(Y ~ ., data = dtrain,
                   num.trees = rfGrid$ntree[1],
                   mtry = rfGrid$mtry[1],
                   min.node.size = rfGrid$node_size[1],
                   sample.fraction = rfGrid$sample_size[1],
                   importance = 'impurity')

# Prediction  
rf.Pred <- predict(rf.Final, dtest)


# 2-Fold Cross-Validation 
# Train Control 
Control <- trainControl(method = 'cv', number = 2,
                        allowParallel = TRUE, verboseIter = TRUE, returnData = TRUE)


# XGBoost --------------------------------------------------------------------------------
# Tuning Grid
xgbGrid <- expand.grid(nrounds = c(300, 500), max_depth = c(5, 10),
                       colsample_bytree = c(0.5, 0.9), eta = c(0.01, 0.1, 0.3),
                       gamma = 0, min_child_weight = 1, subsample = 1)

# Iteration
set.seed(100)
cv.xgb <- train(Y ~ ., data = dtrain,
                trControl = Control, tuneGrid = xgbGrid, method = 'xgbTree')

params <- cv.xgb$finalModel$params

# Final model using the best parameters
xgb.Final <- xgboost(data = Xtrain_f, label = Ytrain,
                     params = params, nrounds = cv.xgb$bestTune$nrounds)

# Prediction    
xgb.Pred <- predict(xgb.Final, Xtest_f, reshape = TRUE)
xgb.Pred <- apply(xgb.Pred, 1, which.max) -1


# KNN -------------------------------------------------------------------------------------
knnGrid <- expand.grid(k = 1:5)

set.seed(100)
cv.knn <- train(Y ~ ., data = dtrain, method = 'knn',
                tuneGrid = knnGrid, trControl = Control)

# Prediction  
knn.Pred <- knn(dtrain[,-1], dtest[,-1], dtrain$Y, k = 3, algorithm = 'kd_tree')


# 3. Evaluation of all models -------------------------------------------------------------
cnn1 <- load_model_hdf5('model-example.hdf5')
cnn2 <- load_model_hdf5('model-nodrop.hdf5')
cnn1.Pred <- cnn1 %>% predcict_classes(Xtest)
cnn2.Pred <- cnn2 %>% predcict_classes(Xtest)

result <- data.frame(CNN = cnn1.Pred, CNN_BASIC = cnn2.Pred, 
                     RF = rf.Pred, XGB = xgb.Pred, KNN = knn3.Pred)

error.all <- apply(result, 2, function(pred) round(mean(dtest$Y == pred), 4)*100)
error.tb <- data.frame(model = names(error.all), acc = error.all, stringsAsFactors = F)
error.tb <- error.tb %>% mutate(error = round(100-acc, 2))

# Figure 8
ggplot(error.tb) + theme_light() + 
  geom_col(aes(fct_reorder(model, error), error, fill = model), color = 'gray50') +
  geom_text(aes(fct_reorder(model, error), error-0.2, label = error), size = 5) +
  scale_fill_brewer(palette = 'Pastel1') + 
  labs(x = '', y = 'Error (%)', title = 'Test Error Rates') + 
  scale_x_discrete(label = c(CNN_BASIC = 'CNN\n(without Dropout)', KNN = '3-NN')) + 
  theme(legend.position = '', axis.text = element_text(size = 12), 
        plot.title = element_text(size = 16, hjust = 0.5), panel.grid = element_blank())

