bike <- vroom("./train.csv")

install.packages("tidymodels")
library(tidymodels)
library(tidyverse)

bike %>% 
  mutate(weather = ifelse(weather==4, 3,weather))

myRecipe <- recipe(count~., data = bike) %>% 
  step_rm(casual, registered) %>% 
  step_time(datetime, features=c("hour", "minute"))
prepped_recipe<- prep(myRecipe)
bake(prepped_recipe, bike)










