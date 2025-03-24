import streamlit as st
import pandas as pd
from ensemble import Ensemble

# Streamlit App
def main():
    # App Title
    st.title("📈 Sales Forecasting Dashboard")
    st.markdown("### Forecasting April Sales using an Ensemble Model")
    st.write("This app predicts sales using a combination of **Moving Average** and **Time Series** models.")

    # Load Data
    data = pd.read_csv("final_data.csv")

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
            final_forecast['Brand'] = final_forecast.Brand.astype(str)

            # Display Final Forecast
            st.success("✅ Model Execution Completed!")
            st.markdown("### 📊 Final Sales Forecast for April")
            st.dataframe(final_forecast.style.format({"Forecast": "{:,.2f}"}))  # Beautify table with formatted numbers

if __name__ == "__main__":
    main()
