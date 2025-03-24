import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import logging

class ML():
    def __init__(self, data):
        self.data = data

    def timeseries(self):
        """
        Calculate april forecast using time series model
        """
        forecast = []
        for index , row in self.data.iterrows():
            sales_df = pd.DataFrame({
                'ds': ['2021-01-01', '2021-02-01', '2021-03-01'], # Dates
                'y': [row['Jan_Sale'], row['Feb_Sale'], row['Mar_Sale']] # Sales
            })
            logging.info(f"Sales data for {row['Brand']} is {sales_df}")
            model = Prophet()
            model.fit(sales_df)
            future_dates = pd.DataFrame({'ds': ['2021-04-01']})
            forecast_df = model.predict(future_dates)  
            logging.info(f"{forecast_df}")
            logging.info(f"Forecast for {row['Brand']} is {forecast_df['yhat'].values[0]}")
            forecast.append(forecast_df['yhat'].values[0])
        self.data['April_Forecast'] = forecast
        ml_forecast = self.data[['Brand', 'April_Forecast']]
        return ml_forecast


    def arima(self):
        """
        Calculate april forecast using ARIMA model
        """
        forecast = []
        for index, row in self.data.iterrows():
            arima_series = [row["Jan_Sale"], row["Feb_Sale"], row["March_Sale"]]
            model = ARIMA(arima_series, order=(1,1,0))
            model_fit = model.fit()
            forecasted_value = model_fit.forecast(steps=1)
            forecast.append(forecasted_value[0])
        self.data['April_Forecast'] = forecast
        arima_forecast = self.data[['Brand', 'April_Forecast']]
        return arima_forecast

    
