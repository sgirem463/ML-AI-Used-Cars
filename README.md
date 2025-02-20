# ML-AI-Used-Cars

## Overview
The project is to apply data analysis and machine learning techniques on a set of used car data, understand what factors decide used car price and come up with a set of guidelines for used car dealers.

I followed the CRISP-DM Framework for this project, the following sections are stages of the frameworks. The notes/guidelines for car dealers can be found in the Deployment section.

## Business Understanding
reads the coupons data into a DataFrame, examine the data and does some cleaning up by removing the 'car' column which isn't relevant to our analysis, also removes rows with NaN values.

### Data Understanding
I went over the data and did the following:

    - checked the geberal info and look at samples of the data
    - used the pandas value_counts(), min() and max() functions to examine values of each column
    - used pandas query to check outliers, e.g. price > 1000,000 and odometer > 300,000


## Data Preparation
I went through Data preparation <-> Modeling three times, to corret problems and refine the model.

My first round of modeling was done before checking closely on the value range of price, I got very wild price predictions with maybe 30% of them in negative, e.g.:

   [55349.1997819 , 131630.43812658, -29487.2160868 ,   1256.34641751,
   -30364.93535473,  46489.62793431, -46423.01999251, 210653.4568858 ,
    58709.43448659, 253703.29984976,  84262.17928232, 126398.9523304 ,
   -35291.93552541,  -2878.65294636,  29698.68876195,  33888.36060247,
    -3117.20475549, -47344.69652718,  70756.80420065, -47416.07640295]
I then found a car with a price of 3736928711, 3.7 billion !!, this car alone can skew the prediction in a big way since its price is more than all other cars I have in my data (about 120,000 for that round). I need to limit the price to a reasonable range. I aslo found some outliers for the odometer column, e.g. 10,000,000 and 9,999,999.

After the 2nd round of modelling I found many 20-years old 200,000-miles cars to have "like new" consition, that cannot be true. The condition column seems to be a complete subjective description from previous owners/sellers. This triggered another round of data preparation by removing the condition column.

At the end I did the followings to prepare the data:

removed columns that are not useful or contain many NaN, e.g. id, size, VIN, condition...
removed outliers by selecting samples with price between 2000 and 100,000.
removed outliers by selection samples with odometer < 200,000
changed year to age with a simple conversion
droped brands that have less than 100 samples
drop samples strange contents, e.g. tramission = "other", cylinder = "other"

## Modeling
In this stage I used Linear Regression, Sequential Feature Selection(5, 10, 20), Ridge and Lasso to model the project.
GridSearchCV() was used for Ridge and Lasso.

## Evaluation
I compared all models tried in the modeling stage, the Ridge model was found to be the best overall. The train and test MSE are as the following:

model   train MSE       test MSE

---------------------------------------

Linear  4.017385e+07    4.057258e+07

SFS-5   5.425768e+07    5.507502e+07

SFS-10  4.664791e+07    4.692533e+07

SFS-20  4.341264e+07    4.355135e+07

ridge   4.018845e+07    4.048460e+07

lasso   4.044163e+07    4.071613e+07


## Deployment
I chose the Ridge model for deployment, some notes for building used car inventory:
1. Never rely on owner/seller description of the car's condition, those are too subjective, one could say their 20 years old 180,000 miles car is "like new" which cannot be true. Similarly you should not believe a 1-year old 2000 miles car is like new either.
2. odometer reading is more important than age, a 5-years old 20,000 miles car is probably more valuable than a 2-years old 100,000 miles car.
3. Powerful and larger cars are more valuable which should be easy to understand
4. Pickup and trucks have good resale values as people don't seem to mind buying used ones.
5. Used sedans and wagons are more and more out of favor, don't get too many of them and don't pay too much for them.
6. Paint color does make a difference, custom color has the highest value and black/silver/white/grey also hold value well, don't pay too much for other colors.
7. Some brands should be avoid, e.g. Fiat, chrysler, chevolet and dodge

## Final Thoughts
As we see in the pridiction vs actural price comparison of the ridge model, some samples have big difference. One hypothesis is this comes from the situation that no accurate condition description can be relied on. Two cars both at 3-years old and 30,000 miles can have very different conditions, one could be worth $25,000 while the other might be $15,000.


#### The Jupyter Notebook is: [prompt_II.ipynb](https://github.com/sgirem463/ML-AI-Used-Cars/blob/main/prompt_II.ipynb)
