setwd('C:/Users/HK/Desktop/CNN/IMDB')

library(tidyverse)
library(keras)
library(glue)
library(imager)



##### Data Preparation #####
  max_features <- 5000
  maxlen <- 100
  lastlen <- 25

  imdb <- dataset_imdb(num_words = max_features)
  word_index <- dataset_imdb_word_index()
  
  xtest <- imdb$test$x %>% pad_sequences(maxlen = maxlen)
  ytest <- imdb$test$y

# dictionary  
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

  
  
##### Functions ##### 

# 개별 단어 찾기    
  find_word <- function(num) {
    word_index_df %>% 
      filter(idx == num) %>% select(word) %>% unlist
  }

# 저장한 활성도 파일 로드   
  getfile <- function(i, y) paste0(glue('CAM/Activation/heat_{i}'), glue('_{y}.csv'))
# 활성도 그림 저장  
  savefile <- function(i) glue('CAM/Plot/cam_{i}.png')

  
  
##### 단어별 활성도 그림 #####
  
  drawheat <- function(i) { # (test 데이터에서 i번째)
    
    series <- xtest[i,] %>% unlist
    y <- ytest[i]
    
    keyword <- NULL
    for (j in 1:maxlen) keyword[j] <- find_word(series[j])
    keyword <- factor(keyword, unique(keyword))
    
    heat <- read.csv(getfile(i, y), header = F) %>% unlist %>% as.numeric
    fitted <- heat[1]
    heat_df <- data.frame(heat = heat[-1])
    heat_img <- as.cimg(heat_df$heat, x = lastlen, y = 1, z = 1)
    
    tmp <- resize(heat_img, size_x = maxlen, size_y = 1, interpolation_type = 3)
    
    heat_df <- as.data.frame(tmp) %>% mutate(keyword = factor(keyword)) %>%
      filter(!(keyword %in% word_index_df$word[1:4])) %>%
      group_by(keyword) %>% summarise(value = max(value))
  
    ggplot(heat_df) + theme_bw() +
      theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
            axis.text.x = element_text(hjust = 1, angle = 45, size = 12), legend.position = '') + 
      scale_fill_viridis_c() + 
      geom_col(aes(x = keyword, y = value, fill = value)) +
      labs(x = 'keyword', title = paste('Grad-CAM of IMDB Test Sample', i),
           subtitle = paste0('Predicted ', fitted, ' & True ', y))
  
  }

  i <- 1
  png(savename(i), 1200, 200) ; drawheat(i) ; dev.off() ; i <- i +1
