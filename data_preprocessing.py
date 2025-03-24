import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging


class data_prep():

    def __init__(self,data,category):
        self.data = data
        self.category = category

    def generate_data(self):
        days_in_q1 = (datetime(2025,4,1) - datetime(2025,1,1)).days
        self.data['purchase_date'] = [datetime(2025,1,1) + timedelta(days=random.randint(1, days_in_q1 - 1)) for _ in range(len(data))]
        _lvl_data = (
            data.groupby([category, "purchase_date"])["Purchase"]
            .sum()
            .reset_index()
            .rename(columns={"purchase_date": "Date", "Purchase": "Total_Sale"})
        )
        return _lvl_data

    def get_monthly_data(self):
        category_lvl_data = self.generate_data()
        category_lvl_data['Month'] = category_lvl_data['Date'].dt.strftime('%b_%Y')
        print(category_lvl_data.head())

        monthly_sales = (
            category_lvl_data.groupby([category,'Month'])['Total_Sale']
            .sum()
            .unstack()
            .reset_index()
        
        )
        monthly_sales.columns = [col.split('_')[0] + '_Sale' if '_2025' in col else col 
                            for col in monthly_sales.columns]
        

        # print(monthly_sales.head())
        logging.info(f"Monthly sales data: {monthly_sales.head()}")
        return monthly_sales



# if __name__ == "__main__":
#     data = pd.read_csv("walmart.csv")
#     category = "Product_Category"
#     data_preparation = data_prep(data,category)
#     updated_data = data_preparation.get_monthly_data()
#     print(updated_data.head())
#     print("Data Preparation Done")
    




