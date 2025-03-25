# Time-series-forecasting

We are forecasting the sale for each brand for the month of april 2025 based on the sale of last 3 months

1. Kaggle dataset : https://www.kaggle.com/datasets/devarajv88/walmart-sales-dataset/data

## 1. Data preprocessing 
1. Add a date column as the dataset doesnt have a dedicated date column
2. There are 3 columns by which we can group by and do the analytics
3. we calculate the monthly sale based on the aggregation provided by the user 

## 2. Moving Average
1. We intialize a set of weights at the start with January having 0.2 , February 0.3 and March 0.5
2. We calculate the log loss and use Gradient Descent which learns and tries to reduce the log loss updating the weights along the way
3. Once the final iteration is done we calculate the final moving average which we are using for the ensemble model


## 3. ML Algorithms 
1. We are giving the user 3 options to choose . 
    a. Prophet - Predicts April sales using Facebook's Prophet algorithm for univariate time series forecasting.
    b. Arima - Forecasts April sales using AutoRegressive Integrated Moving Average (ARIMA).
    c. Kmeans - Forecasts sales by grouping similar patterns using K-Means clustering.


## 4. ENSEMBLE Model 
1. The user can determine how much the want each the moving average and ML to contribute 
2. Ensemble model combines predictions from machine learning (ML) and moving average (MA) models using weighted averaging to produce more robust sales forecasts.


