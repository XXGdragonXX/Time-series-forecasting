import streamlit as st
import pandas as pd
from ensemble import Ensemble

# Streamlit App
def main():
    # App Title
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("### Forecasting April Sales using an Ensemble Model")
    st.write("This app predicts sales using a combination of **Moving Average** and **Time Series** models.")
    st.write("The final forecast is calculated using an **Ensemble Model** that combines both predictions.")
    st.write("Use the sidebar to adjust the model weightage and click the button to run the model.")
    st.write("Dataset link : https://www.kaggle.com/datasets/devarajv88/walmart-sales-dataset/data ")
    st.write("SQL Query to add Date column : ") 
    st.code("""
    UPDATE purchase_data
    SET purchase_date = DATE_ADD('2025-01-01', INTERVAL FLOOR(1 + (RAND() * 90)) DAY);
    """, language="sql")

    st.write("SQL Query to create monthly sale for each brand : ")
    st.code("""
        CREATE VIEW purchase_lvl_data AS 
        select 
        pd.Product_Category as `Brand`,
        pd.purchase_date as `Date`,
        sum(pd.Purchase) as `Total_Sale`
        from purchase_data pd 
        GROUP BY 
        pd.Product_Category,
        pd.purchase_date;

        -- Create the second view: monthly_sale_jan

        CREATE VIEW monthly_sale_jan AS
        select 
        Brand,
        sum(Total_Sale) as `Jan_Sale`
        from purchase_lvl_data
        WHERE 
        MONTH(Date) = 1
        GROUP BY Brand;

        -- create veiw for sale for february
        CREATE VIEW monthly_sale_feb AS
        select 
        Brand,
        sum(Total_Sale) as `feb_Sale`
        from purchase_lvl_data
        WHERE 
        MONTH(Date) = 2
        GROUP BY Brand;

        -- create veiw for sale for March
        CREATE VIEW monthly_sale_march AS
        select 
        Brand,
        sum(Total_Sale) as `March_Sale`
        from purchase_lvl_data
        WHERE 
        MONTH(Date) = 2
        GROUP BY Brand;


        -- Create the final table
        CREATE TABLE FINAL_TABLE 
        SELECT 
        jan.Brand as `Brand`,
        jan.Jan_Sale as `Jan_Sale`,
        feb.feb_Sale as `Feb_SALE`,
        march.March_Sale as `March_Sale`

        from 
        monthly_sale_jan jan
        JOIN
        monthly_sale_feb feb
        ON
        jan.Brand = feb.Brand
        JOIN
        monthly_sale_march march 
        ON 
        jan.Brand = march.Brand;

    """, language="sql")
    # Load Data
    data = pd.read_csv("final_data.csv")
    data['Brand'] = data['Brand'].astype(str)  # Convert Brand to string for better display
    # Display Data Overview
    with st.expander("🔍 View Dataset Overview"):
        st.write("#### Columns in the Dataset:")
        st.write(data.columns.tolist())  # Display as a list
        st.write("#### Unique Brands:")
        st.write(", ".join(map(str, data['Brand'].unique())))  # Display brands in a single line

    # Sidebar for Model Weightage Selection
    st.sidebar.header("⚖️ Model Weightage Selection")

    # ML Model Weight Slider (0 to 1)
    weight_ml = st.sidebar.slider("ML Model Weight (%)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # Ensure sum = 1
    weight_ma = round(1.0 - weight_ml, 2)

    # Display updated values
    st.sidebar.write(f"🟢 ML Weight: **{weight_ml}**")
    st.sidebar.write(f"🔵 Moving Average Weight: **{weight_ma}** (Auto-Adjusted)")

    # Ensemble Model Explanation
    st.markdown("### 🔗 How the Ensemble Model Works")
    st.write("The **final forecast** is computed as a weighted combination of:")
    st.write(f"- **ML Model Prediction** (Weight: **{weight_ml}**)")
    st.write(f"- **Moving Average Model Prediction** (Weight: **{weight_ma}**)")
    st.code("""
    final_forecast = (weight_ml * forecast_ml) + (weight_ma * forecast_ma)
    """, language="python")

    # Run Model Button
    if st.button("🚀 Run Ensemble Model"):
        with st.spinner("Running Ensemble Model... Please wait. ⏳"):
            ensemble = Ensemble(data, weight_ml, weight_ma)
            final_forecast = ensemble.final_forecast()

            # Convert Brand to string for better display
            # final_forecast['Brand'] = final_forecast.Brand.astype(str)

            final_table = pd.join(data,final_forecast, on = 'Brand')

            # Display Final Forecast
            st.success("✅ Model Execution Completed!")
            st.markdown("### 📊 Final Sales Forecast for April")
            st.dataframe(final_table.style.format({"Forecast": "{:,.2f}"}))  # Beautify table with formatted numbers

if __name__ == "__main__":
    main()
