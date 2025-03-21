import pandas as pd
import numpy as np
import logging

class MA():
    def __init__(self, data):
        self.data = data
        self.weight_jan = 0.2
        self.weight_feb = 0.3
        self.weight_mar = 0.5
        self.learning_rate = 0.01
        self.tolerance = 0.0001
    def calculate_moving_average(self):
        """
        we are calculating based on the sliding window moving average 

        """
        MA = []
        for index, row in self.data.iterrows():
            month_dict = {
                "Brand": row['Brand'],
                "Forecast": self.weight_jan * row['Jan_Sale'] + self.weight_feb * row['Feb_Sale'] + self.weight_mar * row['March_Sale']
            }
            MA.append(month_dict)
        df_MA = pd.DataFrame(MA)    
        return df_MA

    def calculate_error(self, forecast_df):
        """
        Calculate the Mean Squared Error (MSE).
        """
        actual = self.data['March_Sale']
        forecast = forecast_df['Forecast']
        mse = np.mean((actual - forecast) ** 2)
        return mse
    
    def gradient_descent(self, max_iterations=50):
        """
        Calculate the gradient descent.
        """
        forecast_df = self.calculate_moving_average()
        iteration = 0
        prev_error = float('inf')

        while iteration < max_iterations:
            # Calculate the forecast
            forecast_df = self.calculate_moving_average()

            # Calculate the error
            error = self.calculate_error(forecast_df)

            # Check for convergence
            if abs(prev_error - error) < self.tolerance:
                print(f"Converged at iteration {iteration} with error {error}")
                break

            # Calculate gradients
            actual = self.data['March_Sale']
            forecast = forecast_df['Forecast']
            error_gradient = 2 * (forecast - actual) / len(actual)

            gradient_jan = np.mean(error_gradient * self.data['Jan_Sale'])
            gradient_feb = np.mean(error_gradient * self.data['Feb_Sale'])
            gradient_mar = np.mean(error_gradient * self.data['March_Sale'])

            # Update weights
            self.weight_jan -= self.learning_rate * gradient_jan
            self.weight_feb -= self.learning_rate * gradient_feb
            self.weight_mar -= self.learning_rate * gradient_mar

            # Ensure weights sum to 1
            total_weight = self.weight_jan + self.weight_feb + self.weight_mar
            self.weight_jan /= total_weight
            self.weight_feb /= total_weight
            self.weight_mar /= total_weight

            # Update previous error
            prev_error = error
            iteration += 1

        print(f"Final weights: Jan={self.weight_jan}, Feb={self.weight_feb}, Mar={self.weight_mar}")
        print(f"Final error: {prev_error}")

        
 
 










csv_data = pd.read_csv('final_data.csv')
MA_final = MA(csv_data)
"""
Calculating the most optimal weights for the moving average model.
"""
MA_final.gradient_descent()

"""
Calculating the moving average.
"""
finalData = MA_final.calculate_moving_average()
print(finalData.head())