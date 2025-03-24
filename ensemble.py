from moving_average import MA
from ML import ML
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging



class Ensemble():
    def __init__(self, data , weight_ml,weight_ma,model,category):
        self.data = data
        self.category = category
        self.ml = ML(data,category)
        self.ma = MA(data,category)
        self.weight_ml = weight_ml
        self.weight_ma = weight_ma
        self.model = model

    def final_forecast(self):
        """
        Calculate the final forecast using ensemble model.
        """
        if self.model == 'Prophet':
            forecast_ml = self.ml.timeseries()
        elif self.model == 'ARIMA':
            forecast_ml = self.ml.arima()
        forecast_ma = self.ma.calculate_moving_average()
        forecast = []
        for index, row in forecast_ml.iterrows():
            month_dict = {
                self.category: row[self.category],
                "April_Forecast": self.weight_ml * row['April_Forecast'] + self.weight_ma * forecast_ma.loc[forecast_ma[self.category] == row[self.category], 'April_Forecast'].values[0]
            }
            forecast.append(month_dict)
        df_forecast = pd.DataFrame(forecast)
        max_clusters = 10
        wcss = []
        for n in range(1, max_clusters+1):
            kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
            deltas = np.diff(wcss)
        deltas2 = np.diff(deltas)
        optimal_clusters = np.argmax(deltas2) + 2 
        

        return df_forecast , optimal_clusters

# if __name__ == "__main__":
#     data = pd.read_csv("data.csv")
#     ensemble = Ensemble(data)
#     final_forecast = ensemble.final_forecast()
#     print(final_forecast.head())