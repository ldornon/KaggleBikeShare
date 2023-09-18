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

install.packages("skimr")
library(skimr)
skimr::skim(bike)

ggplot(data= bike, aes(x=humidity,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)

ggplot(data= bike, aes(x=datetime,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)

bike$weather <- factor(bike$weather, levels = 1:4, labels = c("Sunny", "Mist", "Rain", "Other"))
bike$season <- factor(bike$season, levels = 1:4, labels = c("Spring","Summer","Fall","Winter"))

# Now, create the boxplot
library(ggplot2)

ggplot(data = bike, aes(x = weather, y = count)) +
  geom_boxplot()

ggplot(data = bike, aes(x = season, fill = season))+
  geom_bar()

ggplot(data= bike, aes(x=datetime,y= count)) +
  geom_density(fill = "blue",alpha=.5)

plot_bar(bike)
GGally::ggpairs(bike)

ggplot(data = bike, aes(x = weather, y = count))+
  geom_bar(stat = "identity")+
  xlab("Weather")+
  ylab("Number of Bikes")

ggplot(data = bike, aes(x =atemp, y = count))+
  geom_histogram()


# Plot 1: 
ggplot(data= bike, aes(x=humidity,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)
# Plot 2:
ggplot(data = bike, aes(x = weather, y = count))+
  geom_bar(stat = "identity")+
  xlab("Weather")+
  ylab("Number of Bikes")
# Plot 3:
ggplot(data = bike, aes(x = weather, y = count)) +
  geom_boxplot()
# Plot 4:
ggplot(data = bike, aes(x = season, fill = season))+
  geom_bar()

library(patchwork)
plot1<- ggplot(data= bike, aes(x=humidity,y= count)) +
  geom_point() +
  geom_smooth(se=FALSE)
plot2<- ggplot(data = bike, aes(x = weather, y = count))+
  geom_bar(stat = "identity")+
  xlab("Weather")+
  ylab("Number of Bikes")
plot3<- ggplot(data = bike, aes(x = weather, y = count)) +
  geom_boxplot()
plot4<- ggplot(data = bike, aes(x = season, fill = season))+
  geom_bar()
(plot1 + plot2)/(plot3 + plot4)

