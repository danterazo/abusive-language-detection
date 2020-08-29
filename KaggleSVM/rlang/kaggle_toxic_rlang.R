# setwd("~/Cloud Storage/Google Drive/+Indiana University/X490 - EMNLP Research/LING-X490/abusive-language-detection/KaggleSVM/rlang")
library(dplyr)

# random
random1 <- read.csv("../data/train.kaggle_toxic.random1.csv", header=TRUE)
random2 <- read.csv("../data/train.kaggle_toxic.random2.csv", header=TRUE)
random3 <- read.csv("../data/train.kaggle_toxic.random2.csv", header=TRUE)

# boosted, topic
topic1 <- read.csv("../data/train.kaggle_toxic.topic1.csv", header=TRUE)
topic2 <- read.csv("../data/train.kaggle_toxic.topic2.csv", header=TRUE)
topic3 <- read.csv("../data/train.kaggle_toxic.topic3.csv", header=TRUE)

# boosted, wordbank
wordbank1 <- read.csv("../data/train.kaggle_toxic.wordbank1.csv", header=TRUE)
wordbank2 <- read.csv("../data/train.kaggle_toxic.wordbank2.csv", header=TRUE)
wordbank3 <- read.csv("../data/train.kaggle_toxic.wordbank3.csv", header=TRUE)

# overlap
overlap.random <- inner_join(inner_join(random1, random2), random3)
overlap.wordbank <- inner_join(inner_join(wordbank1, wordbank2), wordbank3)
