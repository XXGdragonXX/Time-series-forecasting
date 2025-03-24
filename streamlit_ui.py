import streamlit as st
import pandas as pd
from ensemble import Ensemble



def main():
    st.title("Sales Forecasting")
    st.write("This app forecasts the sales for April using ensemble model.")
    data = pd.read_csv("final_data.csv")
    st.write("Columns:")
    st.write(data.columns)
    st.write("Brands")
    st.write(data['Brand'].unique())
    weight_ml = st.input("Enter the weightage for ML model")
    weight_ma = st.input("Enter the weightage for Moving Average model")


    st.write("Ensemble Model:")
    st.write("The ensemble model uses the moving average and time series model to forecast the sales for April.")
    st.write("The final forecast is calculated using the weighted average of the forecasts from the two models.")
    st.write("The weights for the two models are set on the values given in the above section.")
    st.write("The final forecast is calculated as follows:")
    st.code("""
    final_forecast = weight_ml * forecast_ml + weight_ma * forecast_ma
    """)
    if st.button("Run Ensemble Model"):
        ensemble = Ensemble(data, weight_ml, weight_ma)
        final_forecast = ensemble.final_forecast()
        final_forecast['Brand'] = final_forecast.Brand.astype(str)
        st.write("The final forecast is then displayed in the table below:")
        st.write("Final Forecast:")
        st.table(final_forecast)
    
if __name__ == "__main__":
    main()