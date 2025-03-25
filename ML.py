import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ML():
    def __init__(self, data ,category):
        self.data = data
        self.category = category

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
            logging.info(f"Sales data for {row[self.category]} is {sales_df}")
            model = Prophet()
            model.fit(sales_df)
            future_dates = pd.DataFrame({'ds': ['2021-04-01']})
            forecast_df = model.predict(future_dates)  
            logging.info(f"{forecast_df}")
            logging.info(f"Forecast for {row[self.category]} is {forecast_df['yhat'].values[0]}")
            forecast.append(forecast_df['yhat'].values[0])
        self.data['April_Forecast'] = forecast
        ml_forecast = self.data[[self.category, 'April_Forecast']]
        return ml_forecast


    def arima(self):
        """
        Calculate april forecast using ARIMA model
        """
        forecast = []
        for index, row in self.data.iterrows():
            arima_series = [row["Jan_Sale"], row["Feb_Sale"], row["Mar_Sale"]]
            model = ARIMA(arima_series, order=(1,1,0))
            model_fit = model.fit()
            forecasted_value = model_fit.forecast(steps=1)
            forecast.append(forecasted_value[0])
        self.data['April_Forecast'] = forecast
        arima_forecast = self.data[[self.category, 'April_Forecast']]
        return arima_forecast


    # def calculate_wcss(data, max_clusters=10):
    # """Calculate WCSS for different numbers of clusters"""
    #     wcss = []
    #     for n in range(1, max_clusters+1):
    #         kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42)
    #         kmeans.fit(data)
    #         wcss.append(kmeans.inertia_)
    #     return wcss

    
    def unsupervised_model(self):
        """
        Calculate april forecast using unsupervised learnning
        """
        max_clusters = 10
        wcss = []
        features = self.data[['Jan_Sale', 'Feb_Sale', 'Mar_Sale']].copy()
        if features.isnull().values.any():
            features = features.fillna(features.mean())
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        primary_keys = self.data[self.category]
        for n in range(1, max_clusters+1):
            kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)
            deltas = np.diff(wcss)
        deltas2 = np.diff(deltas)
        optimal_clusters = np.argmax(deltas2) + 2 
        logging.info(f"Optimal number of clusters: {optimal_clusters}")
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(scaled_features)

        clustered_data = pd.DataFrame({
            self.category: primary_keys.values, 
            'Cluster': clusters,
            'Jan_Sale': self.data['Jan_Sale'].values,
            'Feb_Sale': self.data['Feb_Sale'].values,
            'Mar_Sale': self.data['Mar_Sale'].values
        })
        cluster_stats = clustered_data.groupby('Cluster').agg({
            'Jan_Sale': 'mean',
            'Feb_Sale': 'mean', 
            'Mar_Sale': 'mean'
        })
        cluster_stats['jan_feb_growth'] = (cluster_stats['Feb_Sale'] - cluster_stats['Jan_Sale']) / cluster_stats['Jan_Sale']
        cluster_stats['feb_mar_growth'] = (cluster_stats['Mar_Sale'] - cluster_stats['Feb_Sale']) / cluster_stats['Feb_Sale']
        cluster_stats['projected_growth'] = cluster_stats['feb_mar_growth'] + cluster_stats['jan_feb_growth']
        clustered_data = clustered_data.merge(cluster_stats['projected_growth'], on='Cluster', how='left')
        clustered_data['April_Forecast'] = clustered_data['Mar_Sale'] * (1 + clustered_data['projected_growth'])
        ml_forecast = clustered_data[[self.category, 'April_Forecast']]
        return ml_forecast
        # return cluster_stats

    def 




