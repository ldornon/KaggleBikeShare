install.packages("tidymodels")
install.packages("poissonreg")
install.packages("stacks")
library(tidymodels)
library(tidyverse)
library(poissonreg)
library(dplyr)
library(vroom)
library(rpart)
library(stacks)


bike_train <- vroom("./train.csv") %>% 
  select(-c("casual", "registered"))
bike_test <-vroom("./test.csv") 




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

##### Log Linear Model

#Transform to log(count)
logBikeTrain <- bike_train %>% 
  mutate(count = log(count))

#Define the model
lin_model <- linear_reg() %>% 
  set_engine("lm")

#Set up workflow
log_lin_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(lin_model) %>% 
  fit(data = logBikeTrain)

#Get preds for test set
log_lin_preds <- predict(log_lin_workflow, new_data = bike_test) %>% 
  mutate(.pred = exp(.pred)) %>% 
  bind_cols(.,bike_test) %>% 
  select(datetime,.pred) %>% 
  rename(count= .pred) %>% 
  mutate(count = pmax(0,count)) %>% 
  mutate(datetime= as.character(format(datetime)))

vroom_write(x = log_lin_preds, file="./BikeLogPreds.csv", delim=",")

#Poisson Regression Model

install.packages("poissonreg")
library(poissonreg)

pois_mod <- poisson_reg() %>% 
  set_engine("glm")

bike_log_pois_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = logBikeTrain)

bike_predictions <- predict(bike_log_pois_workflow,new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_predictions, file="./BikeLogPoisPreds.csv", delim=",")

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
  step_mutate(hour = as.factor(datetime_hour)) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(datetime)


preg_model <- linear_reg(penalty=0, 
                         mixture=0) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(preg_model) %>% 
  fit(data = logBikeTrain)

Penalized_regression <- predict(preg_wf, new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=Penalized_regression, file="./BikeLogPenalizedPreds.csv", delim=",")

###############



preg_model <- linear_reg(penalty=tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(preg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)

folds <- vfold_cv(bike_train, v= 5, repeats = 5)

CV_results <- preg_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse,mae,rsq))

collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>% 
  ggplot(data=., aes(x = penalty, y= mean, color = factor(mixture))) +
  geom_line()

bestTune <- CV_results %>% 
  select_best("rmse")

final_wf <-
  preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = logBikeTrain)

bike_predictions <- final_wf %>% 
  predict(new_data = bike_test) %>% 
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_predictions, file="./LogTuningPreds.csv", delim=",")

###########################
#### Regression Trees #####
###########################

install.packages("rpart")
library(rpart)
library(tidymodels)


my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity= tune(),
                        min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

myRecipe <- recipe(count~., data = bike_train) %>% 
  step_mutate(weather = ifelse(weather==4, 3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour")
prepped_recipe<- prep(myRecipe)
bake(prepped_recipe, bike_train)

bike_tree_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(my_mod) 

tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 10)

folds <- vfold_cv(bike_train, v= 5, repeats = 5)


CV_results <- bike_tree_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))


collect_metrics(CV_results) %>% 
  filter(.metric=="rmse")


bestTune <- CV_results %>% 
  select_best("rmse")


final_wf <-
  preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = logBikeTrain)

bike_tree_predictions <- final_wf %>% 
  predict(new_data = bike_test) %>% 
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_tree_predictions, file="./LogRegressionTreePreds.csv", delim=",")




######Random Forest

install.packages("ranger")
library(tidymodels)


my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


bike_rand_forest_workflow <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(my_mod) 


tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(range = c(1,10)),
                            levels=4)

folds <- vfold_cv(bike_train, v= 10, repeats = 1)

CV_results <- bike_rand_forest_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>% 
  ggplot(data=., aes(x = min_n, y= mean, color = factor(min_n))) +
  geom_line()

bestTune <- CV_results %>% 
  select_best("rmse")

final_wf <-
  bike_rand_forest_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = logBikeTrain)

bike_rand_forest_preds <- final_wf %>% 
  predict(new_data = bike_test) %>% 
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_rand_forest_preds, file="./NewRandForestBikePreds.csv", delim=",")


##################################
##################################

folds <- vfold_cv(bike_train, v= 5, repeats = 1)


untunedModel<- control_stack_grid()
tunedModel<- control_stack_resamples()

# Penalized Regression Model

myRecipe <- recipe(count~., data = bike_train) %>% 
  step_mutate(weather = ifelse(weather==4, 3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(datetime)

preg_model <- linear_reg(penalty=tune(), 
                         mixture=tune()) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(myRecipe) %>% 
  add_model(preg_model) %>% 
  fit(data = bike_train) 

preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 3)

preg_models <- preg_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)


# Linear Regression Model

lin_reg <- linear_reg() %>% 
  set_engine("lm")

lin_reg_wf <- workflow() %>% 
  add_model(lin_reg_spec) %>% 
  add_recipe(myRecipe)

lin_reg_model <- fit_resamples(
  lin_reg_wf,
  resamples = folds, 
  metrics = metric,
  control = tunedModel
)

###############

NewRecipe <- recipe(count~., data = bike_train) %>% 
  step_mutate(weather = ifelse(weather==4, 3,weather)) %>% 
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>% 
  step_mutate(hour = as.factor(datetime_hour)) %>% 
  step_date(datetime, features= "year") %>% 
  step_rm(datetime)
prepped_recipe<- prep(NewRecipe)
bake(prepped_recipe, bike_train)

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


bike_rand_forest_workflow <- workflow() %>% 
  add_recipe(NewRecipe) %>% 
  add_model(my_mod) 


tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(range = c(1,10)),
                            levels=4)

folds <- vfold_cv(bike_train, v= 10, repeats = 1)

CV_results <- bike_rand_forest_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>% 
  ggplot(data=., aes(x = mtry, y= mean, color = factor(min_n))) +
  geom_point()

bestTune <- CV_results %>% 
  select_best("rmse")

final_wf <-
  bike_rand_forest_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = logBikeTrain)

bike_rand_forest_preds <- final_wf %>% 
  predict(new_data = bike_test) %>% 
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime))) 
vroom_write(x=bike_rand_forest_preds, file="./UpdateRandForestBikePreds.csv", delim=",")




















