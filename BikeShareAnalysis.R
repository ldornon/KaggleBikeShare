bike_train <- vroom("./train.csv") %>% 
  select(-c("casual", "registered"))
bike_test <-vroom("./test.csv") 

install.packages("tidymodels")
library(tidymodels)
library(tidyverse)
library(dplyr)
library(vroom)



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

#Poisson Regression Model

install.packages("poissonreg")
library(poissonreg)

pois_mod <- poisson_reg() %>% 
  set_engine("glm")

bike_pois_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = bike_train)

bike_predictions <- predict(bike_pois_workflow,new_data = bike_test) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_predictions, file="./TestPoisPreds.csv", delim=",")

#Penalized Regression

library(tidymodels)
library(poissonreg)
install.packages("glmnet")

myRecipe <- recipe(count~., data = bike_train) %>% 
  step_mutate(weather = ifelse(weather==4, 3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(datetime)


preg_model <- linear_reg(penalty=0, mixture=0 ) %>% 
  set_engine("glmnet")
preg_wf <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(preg_model) %>% 
  fit(data = bike_train)
Penalized_regression <-predict(preg_wf, new_data = bike_test) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=Penalized_regression, file="./TestPenalizedPreds.csv", delim=",")

###############

preg_model <- linear_reg(penalty=tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(preg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(bike_train, v= 5, repeats = 5)

CV_results <- preg_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics= metric_set(rmse,mae,rsq))

collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>% 
  ggplot(data=., aes(x = penalty, y= mean, color = factor(mixture))) +
  geom_line()

bestTune <- CV_results %>% 
  select_best("rmse")

final_wf <-
  preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = bike_train)

final_wf %>% 
  predict(new_data = bike_test)


























