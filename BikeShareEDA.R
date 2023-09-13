library(tidyverse)
library(vroom)
install.packages("DataExplorer")
library(DataExplorer)
install.packages("GGally")
library(GGally)
library(patchwork)


bike <- vroom("./train.csv")

#Variable type of each column
dplyr::glimpse(bike)
#Visualization of variable types
DataExplorer::plot_intro(bike)

#Histogram of numerical variables
plot_histogram(bike)
#Plots the proportion of values missing for each variable in the dataset
plot_missing(bike)
#Shows possible correlations between variables in the dataset
plot_correlation(bike)

ggplot(data = bike, aes(x=temp, y = count))+
  geom_point()+
  geom_smooth(se=FALSE)

ggplot(data = bike, aes(x=atemp, y = count))+
  geom_point()+
  geom_smooth(se=FALSE)

skimr::skim(bike)
install.packages("skimr")
library(skimr)

ggplot(data= bike, aes(x=datetime,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)

ggplot(data= bike, aes(x=windspeed,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)

ggplot(data= bike, aes(x=datetime,y= count)) +
  geom_density(fill = "blue",alpha=.5)

plot_bar(bike)
GGally::ggpairs(bike)

ggplot(data = bike, aes(x = weather, y = count))+
  geom_bar(stat = "identity")+
  xlab("Weather")+
  ylab("Number of Bikes")





