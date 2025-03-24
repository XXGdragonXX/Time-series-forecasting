import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

dataframe = pd.read_csv("walmart.csv")
dataframe['purchase_date'] = [datetime(2025,1,1) + timedelta(days=random.randint(1, 90)) for _ in range(len(dataframe))]
category = 'User_ID'
def generate_data(data,category):
    _lvl_data = (
        data.groupby([category, "purchase_date"])["Purchase"]
        .sum()
        .reset_index()
        .rename(columns={"purchase_date": "Date", "Purchase": "Total_Sale"})
    )

    return _lvl_data

def get_monthly_data(data,category):
    category_lvl_data = generate_data(data,category)
    print(category_lvl_data.head())
    category_lvl_data['Month'] = category_lvl_data['Date'].dt.strftime('%b_%Y')
    monthly_sales = (
        category_lvl_data.groupby([category,'Month'])['Total_sale']
        .sum()
        .unstack()
        .reset_index()
    
    )
    monthly_sales.columns = [col.split('_')[0] + '_Sale' if '_' in col else col 
                        for col in monthly_sales.columns]
    

    # monthly_sales = {}
    # months = category_lvl_data['date'].dt.month.unique()
    # for month in months:
    #     month_name = pd.to_datetime(f'2025-{month}-01').strftime('%b')  # Convert month number to name
    #     monthly_sales[month_name] = (
    #         category_lvl_data[category_lvl_data['Date'].dt.month == month]
    #         .groupby(category)['Total_Sale']
    #         .sum()
    #         .reset_index()
    #         .rename_columns = {'Total_Sale':f'{month_name}_Sale'}
    #     )
    # final_table = list(monthly_sales.values())[0]
    # for month_df in list(monthly_sales.values())[1:]:
    #     final_table = final_table.merge(month_df, on=category, how="inner")

    print(monthly_sales.head())







get_monthly_data(dataframe,category)
    




