bike_train <- vroom("./train.csv") %>% 
  select(-c("casual", "registered"))
bike_test <-vroom("./test.csv") 

install.packages("tidymodels")
library(tidymodels)
library(tidyverse)
library(dplyr)




myRecipe <- recipe(count~., data = bike_train) %>% 
  step_mutate(weather = ifelse(weather==4, 3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour")
prepped_recipe<- prep(myRecipe)
bake(prepped_recipe, bike_train)
bake(prepped_recipe, new_data = bike_test)

view(bike_train)
view(bike_test)

my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(my_mod) %>% 
  fit(data = bike_train)

extract_fit_engine(bike_workflow) %>%
  summary()


test_preds <- predict(bike_workflow, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
vroom_write(x=test_preds, file="./TestPreds.csv", delim=",")










