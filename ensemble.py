from moving_average import MA
from ML import ML
import pandas as pd
import numpy as np
import logging



class Ensemble():
    def __init__(self, data):
        self.data = data
        self.ml = ML(data)
        self.ma = MA(data)
        self.weight_ml = 0.5
        self.weight_ma = 0.5    

    def final_forecast(self):
        """
        Calculate the final forecast using ensemble model.
        """
        forecast_ml = self.ml.timeseries()
        forecast_ma = self.ma.calculate_moving_average()
        forecast = []
        for index, row in forecast_ml.iterrows():
            month_dict = {
                "Brand": row['Brand'],
                "April_Forecast": self.weight_ml * row['April_Forecast'] + self.weight_ma * forecast_ma.loc[forecast_ma['Brand'] == row['Brand'], 'April_Forecast'].values[0]
            }
            forecast.append(month_dict)
        df_forecast = pd.DataFrame(forecast)
        return df_forecast

# if __name__ == "__main__":
#     data = pd.read_csv("data.csv")
#     ensemble = Ensemble(data)
#     final_forecast = ensemble.final_forecast()
#     print(final_forecast.head())