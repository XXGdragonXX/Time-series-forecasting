import streamlit as st
import pandas as pd
import logging
from ensemble import Ensemble
from data_preprocessing import data_prep

# Streamlit App
def main():
    # App Title
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("### Forecasting April Sales using an Ensemble Model")
    st.write("This app predicts sales using a combination of **Moving Average** and **Time Series** models.")
    st.write("The final forecast is calculated using an **Ensemble Model** that combines both predictions.")
    st.write("Use the sidebar to adjust the model weightage and click the button to run the model.")
    st.write("Dataset link : https://www.kaggle.com/datasets/devarajv88/walmart-sales-dataset/data ")
    # Display Data Overview
 # Sidebar Controls
    st.sidebar.header("⚙️ Configuration")
    
    # Category Selection
    category = st.sidebar.selectbox(
        "Select Category", 
        ["User_ID", "Product_ID", "Product_Category"],
        help="Select the grouping category for sales analysis"
    )

    # Model Selection
    model = st.sidebar.selectbox(
        "Select Forecasting Model",
        ["Prophet", "ARIMA"],
        help="Choose the time series forecasting model"
    )

    # Weight Selection
    st.sidebar.markdown("### ⚖️ Model Weightage")
    weight_ml = st.sidebar.slider(
        "ML Model Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Adjust the weight for the ML model prediction"
    )
    weight_ma = round(1.0 - weight_ml, 2)
    
    st.sidebar.markdown(f"""
    - 🟢 ML Weight: **{weight_ml}**
    - 🔵 Moving Average Weight: **{weight_ma}**
    """)


    # Model Execution
    st.markdown("---")
    st.markdown("### 🚀 Forecast April Sales")
    
    if st.button("Run Ensemble Model", type="primary"):
        data = pd.read_csv("walmart.csv")
        logging.info(f"Data loaded successfully: {data.shape}")
        data_preparation = data_prep(data,category)
        updated_data = data_preparation.main()


        with st.expander("🔍 Dataset Overview"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Columns in Dataset:**")
                st.write(updated_data.columns.tolist())
            with col2:
                st.write("**Unique Values:**")
                st.write(updated_data[category].nunique())


        with st.spinner("Running ensemble model... This may take a few moments"):
            ensemble = Ensemble(updated_data, weight_ml, weight_ma, model,category)
            final_forecast , optimal_clusters = ensemble.final_forecast()
            
            # Display Results
            st.success("Forecast completed successfully!")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(
                    final_forecast.style.format("{:,.2f}"),
                    height=400
                )
            with col2:
                st.metric(
                    "Total April Forecast", 
                    f"${final_forecast['April_Forecast'].sum():,.2f}"
                )
            
            # Visualization
            st.line_chart(
                final_forecast.set_index(category)['April_Forecast']
            )
                
            # except Exception as e:
            #     st.error(f"Model execution failed: {str(e)}")
            st.write(f"Optimal number of clusters for KMeans: {optimal_clusters}")


    # Footer
    st.footer(" .......Made with ❤️ by XXGdragonXX .....")

if __name__ == "__main__":
    main()