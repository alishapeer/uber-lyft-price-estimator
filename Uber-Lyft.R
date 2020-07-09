# CS 555 Term Project - Uber/Lyft Car Rides
# Alisha Peermohamed | Fall 2019. 

## PLEASE READ: I have commented out the data cleaning section of the code. This
## is because the original datasets were too large to upload (~630,000 instances). 
## Attached is the cleaned dataset, 'uber_lyft_dataset.csv', which we will open in
## data visualization portion of the code (line 143). 

# Original Dataset: 
# https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices#cab_rides.csv 

# library(tidyr)
# options(scipen=999) # prevents scientific notation for time
# #setwd('~/Desktop/CS 555/Term Project/')
# 
# # This Dataset includes samples of various app based rides, both on uber and lyft, and
# # information related to that ride. The dataset records the distance, time, destination,
# # cab company (Uber or Lyft), and type of cab used on the ride in Boston. 
# rides_data <- as.data.frame(read.csv('cab_rides.csv', stringsAsFactors = FALSE, header = TRUE))
# 
# ##DATA CLEANING: 
# # Removing unused columns: Destination, Surge Multiplier, ride ID, Product_ID: 
# drop <- c('destination','surge_multiplier', 'id', 'product_id')
# rides_data <- rides_data[ , !(names(rides_data) %in% drop)]
# 
# # Removing rows with missing price value: 
# rides_data <-rides_data %>% drop_na(price)
# 
# # Reducing Dataset size to 500 rows using Systematic Sampling: 
# #Systematic sampling
# N = nrow(rides_data)
# n = 500
# k <- ceiling(N / n)
# r <- sample(k, 1)
# rows <- seq(r, by = k, length = n)
# rides <- rides_data[rows, ]
# 
# ## Adding weather column to rides data
# # Second dataset: Weather data at epoch time and area. Rain column indicates inches of Rain. 
# weather_data <- as.data.frame(read.csv('weather.csv', stringsAsFactors = FALSE, header = TRUE))
# drops <- c('rain', 'pessure', 'humidity', 'wind', 'clouds')
# weather_data <- weather_data[ , !(names(weather_data) %in% drops)]
# 
# #Convert epoch timestamp to Hour of Day:
# # truncating epoch time to 10 digits: 
# for (i in 1:nrow(rides)) {
#   rides[i, 'time_stamp'] = round(rides[i, 'time_stamp']/10^3)
# }
# 
# #Extracting the hour of the day, month, and hour of the ride
# rides$hour <- NA 
# rides$month <- NA
# rides$day <- NA
# for (i in 1:nrow(rides)) {
#   time <-  rides[i, 'time_stamp']
#   z <- as.POSIXlt(time, origin='1970-01-01', tz='EST')
#   hour <- unclass(z)$hour
#   month <- unclass(z)$mon
#   day <- unclass(z)$mday
#   rides[i, 'hour'] = hour
#   rides[i, 'month'] = month
#   rides[i, 'day'] = day
# }
# 
# # Extracting the day, month, and hour of weather recording
# weather_data$hour <- NA 
# weather_data$month <- NA
# weather_data$day <- NA
# for (i in 1:nrow(weather_data)) {
#   time_w <- weather_data[i, 'time_stamp']
#   x <- as.POSIXlt(time_w, origin='1970-01-01', tz='EST')
#   month <- unclass(x)$mon
#   day <- unclass(x)$mday
#   hour <- unclass(x)$hour
#   weather_data[i, 'hour'] = hour
#   weather_data[i, 'month'] = month
#   weather_data[i, 'day'] = day
# }
# 
# #connecting ride with temperature at the time of ride
# rides$temperature <- NA
# for (i in 1:nrow(rides)) {
#   hour <- rides[i, 'hour']
#   month <- rides[i, 'month']
#   day <- rides[i, 'day']
#   location <- rides[i, 'source']
#   temp_data <- subset(weather_data, weather_data$month == rides[i, 'month']
#                       & weather_data$day == rides[i, 'day'] 
#                       & weather_data$hour == rides[i, 'hour']
#                       & weather_data$location == rides[i, 'source'])[1,]
#   rides[i, 'temperature'] = temp_data$temp
# }
# 
# #Removing rows with NA temperature values: 
# rides <-rides %>% drop_na(temperature)
# 
# ## Encoding Ride Type into Numerical Factors: 
# 
# # * Grouped Uber and Lyft Rides by their product levels: 
# #       * (1): UberPool, Shared
# #       * (2): UberX, Lyft, WAV
# #       * (3): UberXL, Lyft XL
# #       * (4): UberBlack, Lux Black, Lux
# #       * (5): UberBlack SUV, Lux Black XL
# 
# for (i in 1:nrow(rides)) {
#   if (rides[i, 'name'] %in% c('UberPool', 'Shared')) {
#     rides[i, 'name'] = 1}
#   if (rides[i, 'name'] %in% c('UberX', 'Lyft', 'WAV')) {
#     rides[i, 'name'] = 2}
#   if (rides[i, 'name'] %in% c('UberXL', 'Lyft XL')) {
#     rides[i, 'name'] = 3}
#   if (rides[i, 'name'] %in% c('Black', 'Lux Black', 'Lux')) {
#     rides[i, 'name'] = 4}
#   if (rides[i, 'name'] %in% c('Black SUV', 'Lux Black XL')) {
#     rides[i, 'name'] = 5}
# }
# 
# ##Final Datacleaning: 
# #   Renaming certain columns
# names(rides)[6] <- 'Product.Level'
# names(rides)[2] <- 'Company'
# names(rides)[1] <- 'Distance'
# names(rides)[4] <- 'Source'
# names(rides)[10] <- 'Temperature'
# names(rides)[5] <- 'Price'
# #   Dropping Unused columns: timestamp, month, day, hour
# drop <- c('time_stamp','month', 'day', 'hour')
# rides <- rides[ , !(names(rides) %in% drop)]
# #   Re-arranging columns to: Company, Source, Product Level, Distance, temperature.
# rides <- rides[,c('Company', 'Source', 'Product.Level', 'Distance', 'Temperature', 'Price')]
# 
# rides_data <-rides_data %>% drop_na(price)
# rides <-rides %>% drop_na(Temperature)
# 
# #setwd('/Users/alishapeermohamed/Desktop/CS 555/Term Project')
# #write.csv(rides, 'uber_lyft_dataset.csv', row.names = FALSE)
# 
# #Removing all variables and data in memory: 
# # remove(list = ls())

#############################################################################################################################
## DATA VISUALIZATION AND ANALYSIS - RESEARCH SCENARIO:

options(scipen=10)
#setwd("~/Desktop/CS 555/Term Project")
rides <- as.data.frame(read.csv('uber_lyft_dataset.csv', stringsAsFactors = FALSE, header = TRUE))

## Research Question: 
# Explore the effects of various factors on the price of the app-sharing ride:
#     * Distance, Pick-up point, Temperature at the time of the ride, and type of product(UberX, Lyft Lux)

## Two Sample Means T-test: 
#       * Testing whether or not the prices of rides leaving from the Financial District are higher than the prices of
#       * of rides leaving from South Station at a 95% confidence level. 

#       * Chose these two areas because I beleive they are the busiest in the city. 

south <- subset(rides, rides$Source == 'South Station')$Price
financial_dis <- subset(rides, rides$Source == 'Financial District')$Price
len_south <- length(south)
len_financial_dis <- length(financial_dis)

#Formal test of Hypothesis: 
#Step1: 
# Null Hypothesis: Prices_south == Prices_financial_district
# Alternate hypothesis: Prices_financial_district > Prices_south
# alpha = 0.05

# Step2: 
# Select the t-statistic as the appropriate test statistic because the standard deviation of the population size is unknown.
# t = (x1bar = x2bar) = (mu1 - mu2) / sqrt((sd1^2/ n1) + (sd2^2/n2))

# Step3: 
sd_south <- sd(south)
sd_fin <- sd(financial_dis)
df <- ((sd_south^2/len_south) + (sd_fin^2/len_financial_dis))^2/(((sd_south^2/len_south)^2/(len_south-1)) + ((sd_fin^2/len_financial_dis)^2/(len_financial_dis-1)))
t_critical <- qt(0.95, df);t_critical #t-critical = 1.66517
# Decision Rule: reject Null hypothesis (H0) if t >= 1.66517
#                Otherwise, do not reject Null hypothesis (H0)

# Step4: 
t.test(financial_dis, south, alternative='greater', conf.level=0.95)
# t-statistic = 1.6719
#mu_south <- mean(south)
#mu_fin <- mean(financial_dis)
#t <- (mu_south - mu_fin) / sqrt((sd_south^2/ len_south) + (sd_fin^2/len_financial_dis))
# The average price of the rides leaving from the Financial district is $20.92 per ride whereas the average
# price of uber rides leaving from South Station is $16.93 per ride. 

#Step5: 
# Reject the Null Hypothesis since the p-value from the T-test is less than the 0.05. 
# We are 95% confident that the mean prices for rides leaving from South station are less than the mean prices of rides leaving from the
# financial district.

#############################################################################################################################
## Correlation Test Between Distance and Price: 

#     * Test to determine whether there is a linear association between the price of a ride and the distance of the ride.
#     * Using samples to determine the price to distance correlation for the entire population. 

r <- cor(rides$Distance, rides$Price); r # r = 0.3641559

# Step1: 
# Null Hypothesis: p_population = 0. There is no linear association between price and distance travelled
# Alternate Hypothesis: p_population =/=. There is a linear association between price and distance travelled. 
# alpha = 0.05 

# Step2: 
# t = r(sqrt((n-2)/(1 - r^2)))

# Step3: 
df <- length(rides$Price) - 2
# associated right-hand probability of alpha/2 = 0.025
t_critical <- qt(0.975, df=df) # t_critical = 1.964758

# Decision Rule: Reject Null Hypothesis if |t| >= 1.964758. 
#                Else: Do not reject Null Hypothesis. 

# Step4:
t <- r*(sqrt((df)/(1 - r^2))); t # t = 8.778396; # p-value < 2.2e-16. 
cor.test(rides$Distance , rides$Price , alternative='two.sided', method='pearson', conf.level = 0.95) 

# Step 5: 
# Reject Null Hypothesis that there is no linear association between confidence level and price. 
# We have significant evidence at the 95% confidence interval that p =/= 0. 
# There is strong evidence of a significant linear association between distance travelled and the price of the ride. 

#############################################################################################################################
## Simple Linear Regression on Distance and Price: 
#     * Developed a SLR predicting the Price of the Ride based on the Distance travelled. 

plot(rides$Distance, rides$Price,
     main = 'Rides Prices for Various Distances Travelled using Uber and Lyft',
     xlab = 'Distance Travelled (in miles)', 
     ylab = 'Price of Ride in Dollars ($)', 
     col = 'darkcyan', cex.main = 1, pch = 1, cex = 0.8) 

SLR <- lm(rides$Price ~ rides$Distance)
summary(SLR)
abline(a=10.402 , b=3.171, col = 'darkorange', lwd = 2)

distance_bar <- mean(rides$Distance)
sd_distance <- sd(rides$Distance)
price_bar <- mean(rides$Price)
sd_price <- sd(rides$Price)

beta1 <- r*sd_price/sd_distance;beta1
beta0 <- price_bar - beta1*distance_bar;beta0

# The equation for the Simple Linear Regression is: Price = 10.3512 + 3.1899(Distance). 

# For every one mile increase in Distance, there is a $3.19 increase in the price. 
# If a person travelled 0 miles, the average price of the ride will be $10.35. 

# Formal inference test involves considering β0,β1 as unknown population
# parameters and determining what we can say about the unknown population 
# parameters given the data we observed from our sample using the ANOVA table. 

anova <- anova(SLR); anova
SE_beta1 <- summary(SLR);SE_beta1
# Test whether there is there a linear relationship between distance travelled and the price of the ride. 

# Step 1: 
# Null Hypothesis[H0]: beta_distance = 0 (There is no linear association)
# Alternate Hypothesis [H1]: beta_distance =/= 0 (There is a linear association)
# alpha = 0.05

# Step 2: 
# F-statistic: F = MS Reg/MS Res with 1 and n-2 = 498 -2 = 496 degrees of freedom

#Step 3: 
# F distribution with 1, 496 degrees of freedom and alpha = 0.05
F_critical <- qf(0.95, df1 = 1, df2 = 496); F_critical # F_critical = 3.8602
# Decision Rule: Reject H0 if F_statistic >= 3.8601, 
#                otherwise can not reject H0. 

#Step 4: 
# Based on the ANOVA table, F-statistic = 77.06

#Step 5: 
# Since the F-statistic > F-critical, 77.06 > 3.8601 and the p-value < 2.2e-16, we reject the Null Hypothesis that
# there is no linear association between the distance travelled and the price of the ride. 
# We have significant evidence at the alpha = 0.05 level that there is a linear association 
# between distance and price. 

# 95% Confidence Interval of Beta_Distance:
beta1_95confidence <- confint(SLR, level = 0.95)[2,];beta1_95confidence
# This means that for every one mile increase in distance, we are 95% confident that the price
# of the ride will increase from $2.48/ride to $3.90/ride. 

r_squared <- r^2; r_squared 
# The adjusted R-squared value is 13.26% of the variation in Price is explained by changes
# in the distance. 

#############################################################################################################################
## Multiple Linear Regression 
#      * Develop a Multiple Linear Regression to explore the effect of distance,
#      * temperature, & product.level together. 

# MLR with Company, Product Level, Temeperature, and Distance as explanatory variables: 
MLR <- lm(rides$Price~rides$Product.Level + rides$Temperature + rides$Distance) 

# Global F-test: Is there a linear relationship between the price of the ride and the distance, temperature, 
# product-level?

# Step1: 
# Null Hypothesis: H0: Beta_distance = Beta_Product_level = Beta_Temperature = 0 
# Distance, Product Level, and Temperature are not predictors of annual salary. 

# Alternate Hypothesis: H1: Beta_distance =/= Beta_Product_level =/= Beta_Temperature =/= 0. 
# At least one in Distance, Product Level, and Temperature is a significant predictor of annual salary. 
# alpha = 0.05 

# Step2: 
# k = 3
# Choose the F-statistic with 3 and 494 degrees of freedom. 

#Step 3: 
qf(.95, df1=3, df2=494) #F(3, 494, 0.05) = F_critical = 2.6229
# Decision Rule: Reject H0 if F >= 2.6229, 
#                Otherwise do not reject the null hypothesis. 

#Step 4: 
summary(MLR)
# F-statistic = 666.9 with p-value < 2.2e-16. 

#Step5: 
# Reject H0 since F-statistic ≥ 2.6229
# We have significant evidence at the α = 0.05 level that Beta_distance =/= 0 
# and/or Beta_Product_level =/= 0 and/or Beta_Temperature =/= 0
# We are 95% confident that there is evidence of a linear association between
# ride price and distance and/or temperature, and/or product level. 

# MLR Inference t-test: 
#       * Test the significance of individual attributes: distance, temperature, and product.level 
#         to guage the relative contribution of each variable at the alpha = 0.05 level. 
#       * Compute the confidence interval for significant variables. 

t_critical <- qt(0.95 , df = 494); t_critical #t_critical = 1.6479
# Decision Rule: Reject H0 if |t| >=1.6479
#                Otherwise do not reject H0

# Testing for Temperature at the alpha = 0.05 level: 
# The t-statistic of the temperature variable is 1.6479 and p-value is 0.0957. We do not have significant
# evidence at the alpha = 0.05 level that temperature has a significant effect on price, after controling
# for other variables. That being said, we are 90% confident that the temperature variable has a significant
# effect on price. For every one degree increase in temperature, there is a $0.05 in the price of the ride. 

# Testing for Distance at the alpha = 0.05 level: 
# The t-statistic of the distance variable is 16.798 and p-value is <2e-16. We have significant
# evidence at the alpha = 0.05 level that distance has a significant effect on price, after 
# controling for other variables. For every one 1 mile increase in distance, the price of the 
# ride increases by $2.93. 

conf_dist <- c(2.9328 - (1.6479*0.17459) , 2.9328 + (1.6479*0.17459)); conf_dist
# dis_95%_confidence_interval: [2.645093, 3.220507]

# Testing for Product.level at the alpha = 0.05 level:
# The t-statistic of the product_level variable is 40.708 and p-value is <2e-16. We have significant
# evidence at the alpha = 0.05 level that product_level has a significant effect on price, after 
# controling for other variables. For every one level increase increase in product level (increasing
# from pool to UberX, or UberX to UberXL), has a $6.04 increase on the price of the ride. 

conf_prodlev <- c(6.03700 - (1.6479*0.14830) , 6.03700 + (1.6479*0.14830)); conf_prodlev
# product_lev_95%_confidence_interval: [5.792616, 6.281384]

# R-squared Value: 
regss <- sum((fitted(MLR) - mean(rides$Price))^2)
resiss <- sum((rides$Price-fitted(MLR))^2)
totalss <- regss + resiss
fstatistic <- (regss/3)/(resiss/494)
pvalue <- 1-pf(fstatistic , df1=2, df2=97)
R2 <- regss/totalss; R2

# The R-squared value for the Multiple Linear Regression is 80.19%. This means that 80.19% of all variation
# in the price of the ride can be explained by variation in distance, product.level, and the temperature outside. 

#############################################################################################################################
## One way ANOVA to compare means across Uber rides and Lyft rides: 
#      * Test the hypothesis that the prices for the population of Uber rides is different from the prices of
#      * the population of Lyft rides

rides$Company <- factor(rides$Company, levels = c('Uber', 'Lyft'))
one_way_ANOVA <- aov(rides$Price~rides$Company , data=rides)

# Global F-test for one-way-ANOVA:

# Null Hypothesis: mu(uber) = mu(lyft). The mean price of Uber rides is the same as the mean price of Lyft rides.
# Alternate hypothesis: mu(uber) =/= mu(lyft). The mean price of Uber rides is not the same as the mean price of Lyft rides.
# alpha = 0.05

# F-statistic with 1 and 498-2 = 496 degrees of freedom  
f_critical <- qf(.95, df1=1, df2=496); f_critical #F_critical = 3.8602
# Decision Rule: Reject Null hypothesis if F-statistic >= 3.8602. 
#                Otherwise, do not reject H0.
summary(one_way_ANOVA)
# F-statistic = 5.73, p-value = 0.017. 
# We reject the Null hypothesis that the mean price of Uber rides is the same as the mean price of Lyft rides. 
# We have significant evidence at the α = 0.05 that there is a difference in prices between Uber rides and Lyft rides. 

#Pairwise Comparison using t-test:

# Since there is only one pair, the t-test will give in the same results at the F-test: 
# Null Hypothesis: mu(uber) = mu(lyft)
# Alternate hypothesis: mu(uber) =/= mu(lyft)
t_critical <- qt (.975 , df =496); t_critical
# alpha = 0.05 | t-statistic with 496 degrees of freedom. 
pairwise.t.test(rides$Price, rides$Company, p.adj='none') 
# p-value(Uber, lyft) = 0.017

# One-Way Anova analysis using a linear regression: 

# The regression model equivalent to the one-way ANOVA model, holding lyft as
# the reference group, is: y = Beta_intercept + Beta_uber(group_uber). 
rides$uber <- ifelse(rides$Company == 'Uber', 1, 0)
rides$lyft <- ifelse(rides$Company == 'Lyft',1,0)

one_way_model <- lm(rides$Price ~ rides$uber, data=rides) 
summary(one_way_model)
# By the fact that our p-value is 0.017, our linear regression confirms that there is
# a significant difference in the price of Uber rides versus lyft rides. 
# The beta_uber is -2.1546. So, we say that average uber price per ride is $2.15 less 
# than the same for lyft rides. 

# Adjusting for other variables (ie. distance, temperature, product_level): 
install.packages("carData")
install.packages("car")
library(carData)
library(car)
adjust_MLR <- lm(rides$Price~rides$Company+rides$Distance + rides$Temperature+rides$Product.Level)
Anova(adjust_MLR, type = 3) 
summary(adjust_MLR)

# After adjusting for other variables (distance, temperature, product level), we can see that although 
# the model passes the Global F-test (indicating that atleast one of the variables does not equal zero), 
# The 'Company' variable does not pass the inference F-test at a alpha = 0.05 level. 
# This is shown by the fact that the p-value of the Company variable is 0.52 and the F-statistic is 0.4145. 
# Thus, after adjusting for other covariants, we are 95% confident that the differences that we saw 
# in the one-way ANOVA model were due to other variable differences across the Uber and Lyft
# as opposed to true differences in Price attributable only to the Company used. 

#Least squares means
install.packages("emmeans")
install.packages('lsmeans')
library(emmeans)
library(lsmeans)
# p-value adjustment: 
emmeans(adjust_MLR, specs = "Company" , contr = "pairwise")

# The least square means (adjusted for distance, temperature, and product.level ) were $17.20 
# per ride and $17.40 per ride for Uber and lyft respectively. We do not have significant evidence
# against the null hypothesis, which is: the price of the uber rides is the same as the price of Lyft rides
# after controlling for other variables in the model. 

## One way ANOVA to compare means across pick-up location: 
#      * Test the hypothesis that the prices for rides significantly vary across pick up location

rides$Source <- factor(rides$Source, levels = unique(rides$Source))
one_way_ANOVA_s <- aov(rides$Price~rides$Source , data=rides)

# Global F-test for one-way-ANOVA:

# Null Hypothesis: mu(pickup points) = mu(pickup oints). The mean price of rides is the same across pick up points. 
# Alternate hypothesis: mu(pick up) =/= mu(pick up). The mean price of rides is not the same across different pick up points. 
# alpha = 0.05

# F-statistic with 12 and 498-12 = 486 degrees of freedom  
f_critical <- qf(.95, df1=12, df2=486); f_critical #F_critical = 1.77211
# Decision Rule: Reject Null hypothesis if F-statistic >= 1.77211
#                Otherwise, do not reject H0.
summary(one_way_ANOVA_s)
# F-statistic = 3.005, p-value = 0.0069.  
# We reject the Null hypothesis that the mean price per ride is the same across pick up locations 
# We have significant evidence at the α = 0.05 that there is a difference in prices based on pick up location. 

#Pairwise Comparison using t-test:

# Null Hypothesis: mu(pick up points) = mu(pick up points) = 0
# Alternate hypothesis: mu(pick up points) =/= mu(pick up points)
# alpha = 0.05 | t-statistic with 486 degrees of freedom. 

aggregate(rides$Price, by=list(rides$Source), summary)
aggregate(rides$Price, by=list(rides$Source), var)
pairwise.t.test(rides$Price, rides$Source, p.adj='none')
TukeyHSD(one_way_ANOVA_s)

# One-Way Anova analysis using a linear regression: 

# The regression model equivalent to the one-way ANOVA model, holding NorthEaster University as
# the reference group, is: y = Beta_intercept + sum(Beta_pickup(group_pickup)). 
rides$NEUni <- ifelse(rides$Source == 'Northeastern University', 1, 0)
rides$North <- ifelse(rides$Source == 'North Station', 1, 0)
rides$Fenway <- ifelse(rides$Source == 'Fenway', 1, 0)
rides$Backbay <- ifelse(rides$Source == 'Back Bay', 1, 0)
rides$BU <- ifelse(rides$Source == 'Boston University', 1, 0)
rides$South <- ifelse(rides$Source == 'South Station', 1, 0)
rides$Beacon <- ifelse(rides$Source == 'Beacon Hill', 1, 0)
rides$Hay <- ifelse(rides$Source == 'Haymarket Square', 1, 0)
rides$WestEnd <- ifelse(rides$Source == 'West End', 1, 0)
rides$NorthEnd <- ifelse(rides$Source == 'North End', 1, 0)
rides$Findist <- ifelse(rides$Source == 'Financial District', 1, 0)
rides$Theatre <- ifelse(rides$Source == 'Theatre District', 1, 0)

# holding Northeastern University as reference group
one_way_model_s <- lm(rides$Price ~ rides$North + rides$Fenway + 
                                    rides$Backbay + rides$BU + rides$South + 
                                    rides$Beacon + rides$Hay + rides$WestEnd
                                    + rides$NorthEnd + rides$Findist + rides$Theatre , data=rides) 
summary(one_way_model_s)
# By the fact that our p-value is 0.00068, our linear regression confirms that there is
# a significant difference in the price of rides across pick up locations.

# Adjusting for other variables (ie. distance, temperature, product_level): 
adjust_MLR_s <- lm(rides$Price~rides$Source+rides$Distance + rides$Temperature+rides$Product.Level)
Anova(adjust_MLR_s, type = 3) 
summary(adjust_MLR_s)

# After adjusting for other variables (distance, temperature, product level), we can see that although 
# the model passes the Global F-test (indicating that atleast one of the variables does not equal zero), 
# The 'Source' variable does not pass the inference F-test at a alpha = 0.05 level. 
# This is shown by the fact that the p-value of the Source variable is 0.7028 and the F-statistic is 0.7363. 
# Thus, after adjusting for other covariants, we are 95% confident that the differences that we saw 
# in the one-way ANOVA model were due to other variable differences across the pick up point
# as opposed to true differences in Price attributable only to the Source used. 

#Least squares means
# p-value adjustment: 
emmeans(adjust_MLR_s, specs = "Source" , contr = "pairwise")

# The least square means (adjusted for distance, temperature, and product.level ) show that rides leaving from
# Boston University have the highest price per ride of $18.40 per ride and rides leaving from Beacon Hill 
# and the West End have the lowest average price per ride of $16.60 per ride. However, we do not have significant evidence
# against the null hypothesis, which is: the price of the uber rides is the same as the price of Lyft rides
# after controlling for other variables in the model. 

####
## One way ANOVA to compare means across Product-Level
#      * Test the hypothesis that the prices for rides significantly vary across praoduct level. 

rides$Product.Level <- factor(rides$Product.Level, levels = unique(rides$Product.Level))
one_way_ANOVA_p <- aov(rides$Price~rides$Product.Level , data=rides)

# Global F-test for one-way-ANOVA:

# Null Hypothesis: mu(product_level) = mu(product_level). The mean price of rides is the same across different product levels.
# Alternate hypothesis: mu(product_level) =/= mu(product_level). The mean price of rides is not the same across different product levels. 
# alpha = 0.05

# F-statistic with 5 and 498-5 = 493 degrees of freedom  
f_critical <- qf(.95, df1=5, df2=493); f_critical #F_critical = 2.23229
# Decision Rule: Reject Null hypothesis if F-statistic >= 2.23229
#                Otherwise, do not reject H0.
summary(one_way_ANOVA_p)
# F-statistic = 330.4, p-value <2e-16.   
# We reject the Null hypothesis that the mean price per ride is the same across product levels.  
# We have significant evidence at the α = 0.05 that there is a difference in prices based on product level. 

#Pairwise Comparison using t-test:

# Null Hypothesis: mu(product_levels) = mu(product_levels) = 0
# Alternate hypothesis: mu(product_levels) =/= mu(product_levels)
# alpha = 0.05 | t-statistic with 493 degrees of freedom. 

aggregate(rides$Price, by=list(rides$Product.Level), summary)
aggregate(rides$Price, by=list(rides$Product.Level), var)
pairwise.t.test(rides$Price, rides$Product.Level, p.adj='none')
TukeyHSD(one_way_ANOVA_p)

# One-Way Anova analysis using a linear regression: 

# The regression model equivalent to the one-way ANOVA model, holding level 1 (UberPool, Lyft Shared) as
# the reference group, is: y = Beta_intercept + sum(Beta_product_level(group_product_level)). 
rides$one <- ifelse(rides$Product.Level == 1, 1, 0)
rides$two <- ifelse(rides$Product.Level == 2, 1, 0)
rides$three <- ifelse(rides$Product.Level == 3, 1, 0)
rides$four <- ifelse(rides$Product.Level == 4, 1, 0)
rides$five <- ifelse(rides$Product.Level == 5, 1, 0)

# holding UberPool as reference group
one_way_model_p <- lm(rides$Price ~ rides$two + rides$three + 
                        rides$four + rides$five , data=rides) 
summary(one_way_model_p)
# By the fact that our p-value is <2e-16, our linear regression confirms that there is
# a significant difference in the price of rides across product_levels. 

# Adjusting for other variables ANCOVA (ie. distance, temperature, product_level): 
adjust_MLR_p <- lm(rides$Price~rides$Product.Level+rides$Distance + rides$Temperature+rides$Product.Level)
Anova(adjust_MLR_p, type = 3) 
summary(adjust_MLR_p)

# After adjusting for other variables (distance, temperature, product level), we can see that although 
# the model passes the Global F-test (indicating that atleast one of the variables does not equal zero), 
# The 'Product Level' variable also passes the inference F-test at a alpha = 0.05 level. 
# This is shown by the fact that the p-value of the product_level variable is <2e-16 and the F-statistic is 547.16. 
# Thus, after adjusting for other covariants, we are 95% confident that Product_level has a significant 
# effect on the price of the ride after controlling for other covariates.

#Least squares means
# p-value adjustment: 
emmeans(adjust_MLR_p, specs = "Product.Level" , contr = "pairwise")

# The least square means (adjusted for distance, temperature, and product.level ) show that level 1 rides have average price
# of $7.56, level 2 rides have average price of $9.75, level 3 rides have average price of
# $16.13 rides, level 4 rides have average price of $20.50, and level 5 rides have average price 
# of $31.52. 

# Combined Final Multi linear regression Model: Evaluating the effects of all our variables:
# Company, Source, Distance, Temperature, Product.Level.  

adjust_MLR_s_c <- lm(rides$Price~rides$Source+rides$Company+rides$Distance + rides$Temperature+rides$Product.Level)
Anova(adjust_MLR_s_c, type = 3) 
summary(adjust_MLR_s_c)

conf_dist <- c(2.83131 - (1.6479*0.18074) , 2.8131 + (1.6479*0.18074)); conf_dist

# After adjusting for other variables (distance, temperature, product level), we can see that although 
# the model passes the Global F-test (indicating that atleast one of the variables does not equal zero), 
# The 'Source' variable does not pass the inference F-test at a alpha = 0.05 level. 
# This is shown by the fact that the p-value of the Source variable is 0.1904 and the F-statistic is 1.3564. 
# Additionally, the Company variable does not pass the inference F-test at the alpha = 0.05 level either
# since the p-value for the Company variable is 0.2045 and the F-statistic is 1.6141. 
# Additionally, the product_level does pass the inference F-test at the alpha = 0.05 level
# since the p-value for the product_level variable is <2e-16 and the F-statistic is 532.7. 

# Thus, after adjusting for other covariants, we are 95% confident that the differences that we saw 
# in the one-way ANOVA models were due to distance and product level as opposed to true differences 
# in Price attributable to the Source, Company used, or temperature. 

# The R-squared value is 84.72% which indicates that 84.72% of the variation in price is due
# model.

#Least squares means
# p-value adjustment: 
prod_level_means <- emmeans(adjust_MLR_s_c, specs = "Product.Level" , contr = "pairwise")

# The least square means (adjusted for distance, temperature, and product.level, company, and source) show 
# that level 1 rides have average price of $7.68, level 2 rides have average price of $9.57, level 3 rides have average price of
# $16.18 rides, level 4 rides have average price of $20.50, and level 5 rides have average price 
# of $31.68. 

