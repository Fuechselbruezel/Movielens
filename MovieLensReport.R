
# Library Setup -----------------------------------------------------------

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if (!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(stringr)
library(caret)
library(kableExtra)
library(recosystem)

# Create edX Dataset ------------------------------------------------------


# Note: this code is part of the course material provided on edX.
# Note: this process could take a couple of minutes

options(timeout = 120)

dl <- "Data/ml-10M100K.zip"
if (!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "Data/ml-10M100K/ratings.dat"
if (!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "Data/ml-10M100K/movies.dat"
if (!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed, ratings_file, movies_file)



# Introduction ------------------------------------------------------

# Set our goal RMSE to later compare it to the models RMSE
goal_rmse <- 0.86490

# Print the head for first insight
head(edx)


# Analysis ------------------------------------------------------
## Exploratory Data Analysis ------------------------------------------------------

# Plot a histogram of rating
hist(edx$rating, main = NULL, xlab = NULL)

# select all movies rated less than 15 times
tmp_ratings <- edx %>% group_by(movieId) %>% summarise(n()) %>% filter(.$`n()` <= 15)

# remove the selected movies
edx <- anti_join(edx, tmp_ratings)

# clear from environment
rm(tmp_ratings)

# create date column in dataset
edx <- edx %>% mutate(date = as_date(as_datetime(timestamp)))

#print first entries
edx %>% select(movieId, userId, timestamp, date) %>% head()

#all available genres
genres <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")

#count appearances for each genre in dataset
list <- list()
for (genre in genres) {
  list[[genre]] <- sum(str_detect(edx$genres, genre))
}

#transform list to long dataframe
pivot_longer(as.data.frame(list), everything(), names_to = "genre", values_to = "count")

#remove from environment
rm(list, genres, genre)

## RMSE ------------------------------------------------------

# this is how we will evaluate our models accuracy
rmse <- function(y_actual, y_predicted) { 
  sqrt(mean((y_actual - y_predicted)^2)) 
}


## Train / Test set ------------------------------------------------------

# using seed for reproducibility
set.seed(1234, sample.kind = "Rounding")

# split dataset into training and test set by 75%, 25% respectively
index <- createDataPartition(y = edx$rating, p = 0.25, list = FALSE)
edx_train <- edx[-index,]
edx_test <- edx[index,]
rm(index)


  
## Modelbuilding ------------------------------------------------------
  
### A Trivial Model ------------------------------------------------------

#calculate rantings mean
mu <- mean(edx_train$rating)

#calculate rmse
rmse_trivial <- rmse(edx_test$rating, mu)

#present results
results <- tibble(Model = "Trivial Model", RMSE = rmse_trivial, Goal = rmse_trivial < goal_rmse)
results


### Movie Effect ------------------------------------------------------

#calculate movie effect
movie_effect <- edx_train %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu))

# calculating the predicted ratings in the test set
fit_movies <- edx_test %>% 
  left_join(movie_effect, by = "movieId") %>% 
  mutate( predict = mu + b_i) %>% 
  pull(predict)

# remove any unknown values
fit_movies <- if_else(is.na(fit_movies), mu, fit_movies)

#calculate rmse
rmse_movie <- rmse(edx_test$rating, fit_movies)

#add to results table and present
results <- bind_rows(results, tibble(Model = "Movie Effect", RMSE = rmse_movie, Goal = rmse_movie < goal_rmse))
results



### User Effect ------------------------------------------------------

# calculate user effect
user_effect <- edx_train %>% 
  left_join(movie_effect, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating - mu - b_i))

#predict ratings with compensating movie & user effect
fit_users <- edx_test %>% 
  left_join(movie_effect, by = "movieId") %>% 
  left_join(user_effect, by = "userId") %>% 
  mutate(predict = mu + b_i + b_u) %>% 
  pull(predict)

#remove unknown values
fit_users <- if_else(is.na(fit_users), mu, fit_users)

#calculate rmse
rmse_movie_users <- rmse(edx_test$rating, fit_users)

#add to results table and present
results <- bind_rows(results, tibble(Model = "Movie + User Effect", RMSE = rmse_movie_users, Goal = rmse_movie_users < goal_rmse))
results

### Movie & User Regularization ------------------------------------------------------

# lambdas to test
lambdas <- seq(0, 7, 0.1)

#test lambdas
rmse_coll <- sapply(lambdas, function(lambda) {
  
  #calculate regularized movie effect
  b_i_reg <- edx_train %>% 
    group_by(movieId) %>% 
    summarise(b_i_reg = sum(rating - mu) / (n() + lambda))
  
  #calculate regularized user effect
  b_u_reg <- edx_train %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(b_u_reg = sum(rating - mu - b_i_reg) / (n() + lambda))
  
  #calculate predictions and rmse
  edx_test %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(predict = mu + b_i_reg + b_u_reg) %>%
    filter(!is.na(predict)) %>%
    summarise(rmse = rmse(rating, predict)) %>%
    pull(rmse)
})

#get best lambda
lambda <- lambdas[which.min(rmse_coll)]

#plot performance curve
qplot(lambdas, rmse_coll, geom = "line", main = NULL, xlab = "Lambda", ylab = "RMSE")

#add to results table and present
results <- bind_rows(results, tibble(Model = "Regularization", RMSE = min(rmse_coll), Goal = min(rmse_coll) < goal_rmse))
results


## Extending the Model ------------------------------------------------------

### Matrix Factorization ------------------------------------------------------

# using seed for reproducibility
set.seed(123, sample.kind = "Rounding")

#staging train and test set for matrix factorization
train_data <- data_memory(user_index = edx_train$userId, item_index = edx_train$movieId, rating = edx_train$rating)
test_data <- data_memory(user_index = edx_test$userId, item_index = edx_test$movieId)

#initialize recosystem
r <- Reco()

#solve for optimal options - this might take a while
opts <- r$tune(train_data,opts = list(dim = c(20, 30, 40), lrate = c(0.1, 0.2), costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 10))

#present optimal parameters
kable(as.data.frame(opts$min))

#train the algorithm on train date with optimal parameters
r$train(train_data, opts = c(opts$min, nthread = 1, niter = 10))

#predict ratings in testdata
pred_mf = r$predict(test_data, out_memory())

#calculate rmse
rmse_mf <- rmse(pred_mf, edx_test$rating)

#add to results table and present
results <- bind_rows(results, tibble(Model = "Matrix Factorization", RMSE = rmse_mf, Goal = rmse_mf < goal_rmse))
results


# Results ------------------------------------------------------

#present results
kable(results)

#stage final test data for prediction
final_data <- data_memory(user_index = final_holdout_test$userId, item_index = final_holdout_test$movieId)

# predict ratings of final_holdout_set
pred_final = r$predict(final_data, out_memory())

#calculate rmse
rmse_final <- rmse(pred_final, final_holdout_test$rating)

#present final model rmse
kable(tibble(Model = "Matrix Factorization - Final", RMSE = rmse_final))

