import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime

# Forecasting imports
from statsmodels.tsa.arima.model import ARIMA
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

########################################
# llm_utils Functions
########################################

def generate_combined_summary(df: pd.DataFrame) -> str:
    summary = f"**File Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n"
    summary += f"**Columns:** {', '.join(df.columns)}\n\n"
    summary += "**Data Types:**\n" + df.dtypes.to_frame(name="Data Type").to_markdown() + "\n\n"
    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_cols.size > 0:
        summary += "**Statistical Summary for Numeric Columns:**\n" + df[numeric_cols].describe().to_markdown() + "\n\n"
    summary += "**First 5 Rows:**\n" + df.head().to_markdown() + "\n\n"
    return summary

def call_llm(model_name: str, prompt: str) -> str:
    """
    Call a local LLM via the Ollama CLI using the provided prompt.
    Returns the model's response as a string.
    """
    import subprocess, logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Calling model '{model_name}' with prompt (first 50 chars): {prompt[:50]}...")
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60  # seconds
        )
        response = result.stdout.decode('utf-8').strip()
        logging.info("Received response from LLM.")
        return response
    except subprocess.TimeoutExpired:
        logging.error("LLM call timed out.")
        return "Error: LLM call timed out."
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        logging.error(f"LLM call failed: {error_msg}")
        return f"Error: {error_msg}"

def simulate_llm_conversation(data_summary: str, model1: str, model2: str) -> (str, str):
    """
    Simulate a conversation between two LLMs.
    The first model provides an analysis, and the second expands upon that analysis.
    Returns a tuple of responses.
    """
    prompt1 = f"Analyze the following sensor data summary and provide detailed insights:\n{data_summary}"
    response1 = call_llm(model1, prompt1)
    prompt2 = f"Based on the analysis below:\n{response1}\nProvide further detailed suggestions and actionable insights."
    response2 = call_llm(model2, prompt2)
    return response1, response2

########################################
# data_pipeline Function
########################################

def load_and_merge_csvs(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a folder, preprocess them, and merge into a single DataFrame.
    Uses an outer join to preserve all columns.
    """
    import glob
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise ValueError("No CSV files found in the provided folder path.")
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            # Check for a time column, using either 'DateTime' or 'timestamp'
            if 'DateTime' in df.columns:
                time_col = 'DateTime'
            elif 'timestamp' in df.columns:
                time_col = 'timestamp'
            else:
                st.warning(f"{file} does not contain a recognizable time column.")
                time_col = None
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])
            df_list.append(df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
    if not df_list:
        raise ValueError("No valid CSV files were loaded.")
    merged_df = pd.concat(df_list, join='outer', ignore_index=True)
    return merged_df

########################################
# Forecasting Helper Functions
########################################

def forecast_arima(series, order=(1,1,1), steps=10):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit.summary().as_text()

def forecast_prophet(df, date_col, val_col, steps=10, freq='15min'):
    df_prophet = df.rename(columns={date_col: 'ds', val_col: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    return model, forecast

########################################
# Streamlit App UI
########################################

st.set_page_config(page_title="B2Twin Hackathon: Data Exploration & Forecasting", layout="wide")
st.title("B2Twin Hackathon: Enhanced Biosphere 2 Digital Twin Dashboard")
st.markdown("""
This dashboard focuses on extracting **actionable insights** from Biosphere 2 sensor data.  
It integrates data exploration, interactive visualization, and forecasting models (ARIMA/Prophet) to help scientists understand and predict environmental conditions.
""")

# Sidebar: LLM Model & Data Upload
st.sidebar.header("Upload and Settings")
llm_options = ["llama3.2", "mistral", "llama2", "phi"]
llm_model1 = st.sidebar.selectbox("LLM Model for Analysis", options=llm_options, index=0)
llm_model2 = st.sidebar.selectbox("Secondary LLM Model", options=llm_options, index=0)

data_input_method = st.sidebar.radio("Data Input Method", ("Single CSV File", "Folder of CSV Files"))

# Sidebar: Forecasting Model Selection and Parameters
forecast_models = ["ARIMA"]
if prophet_available:
    forecast_models.append("Prophet")
forecast_method = st.sidebar.selectbox("Select Forecasting Model", forecast_models)

########################################
# Load Data
########################################
df = None
if data_input_method == "Single CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors='coerce')
                df = df.dropna(subset=[time_cols[0]])
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
elif data_input_method == "Folder of CSV Files":
    folder_path = st.sidebar.text_input("Enter folder path for CSV files")
    if folder_path:
        try:
            df = load_and_merge_csvs(folder_path)
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors='coerce')
                df = df.dropna(subset=[time_cols[0]])
        except Exception as e:
            st.error(f"Error loading folder: {e}")

########################################
# Main UI: Tabs for Data Exploration, Forecasting, LLM Interpretation
########################################
if df is not None:
    # Sort by time if available
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    if time_cols:
        df = df.sort_values(by=time_cols[0])
    
    st.subheader("Uploaded File Summary")
    combined_summary = generate_combined_summary(df)
    st.markdown(combined_summary)
    
    # Show basic sensor info for context
    sensor_info = {}
    for col in df.select_dtypes(include=['number']).columns:
        sensor_info[col] = {"avg": df[col].mean(), "min": df[col].min(), "max": df[col].max()}
    st.write("### Sensor Data Summary (Averages, Min, Max)")
    st.write(sensor_info)
    
    # Create Tabs: Data Exploration, Forecasting, LLM Interpretation
    tab_explore, tab_forecast, tab_llm = st.tabs(["Data Exploration", "Forecasting", "LLM Interpretation"])
    
    ########################################
    # Tab: Data Exploration
    ########################################
    with tab_explore:
        st.header("Data Exploration & Interactive Visualization")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        time_col = time_cols[0] if time_cols else None
        if numeric_cols:
            chart_type = st.selectbox("Select Chart Type", ["Line Plot", "Scatter Plot", "Histogram", "Box Plot"])
            if chart_type in ["Line Plot", "Scatter Plot"] and time_col:
                col_choice = st.selectbox("Select a numeric column to plot over time", numeric_cols)
                if col_choice:
                    if chart_type == "Line Plot":
                        fig = px.line(df, x=time_col, y=col_choice, title=f"{col_choice} over Time")
                    else:
                        fig = px.scatter(df, x=time_col, y=col_choice, title=f"{col_choice} over Time")
                    fig.update_layout(xaxis_title="Time", yaxis_title=col_choice, xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                col_choice = st.selectbox("Select a numeric column", numeric_cols)
                if col_choice:
                    if chart_type == "Histogram":
                        fig = px.histogram(df, x=col_choice, nbins=30, title=f"Histogram of {col_choice}")
                    elif chart_type == "Box Plot":
                        fig = px.box(df, y=col_choice, title=f"Box Plot of {col_choice}")
                    fig.update_layout(xaxis_title=col_choice, yaxis_title="Frequency")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for visualization.")

    ########################################
    # Tab: Forecasting
    ########################################
    with tab_forecast:
        st.header("Forecasting Models")
        st.markdown("Select a numeric column to forecast (e.g., CO2_desert[ppm]). The model will use default settings derived from the data.")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        forecast_column = st.selectbox("Select column to forecast", numeric_cols, index=numeric_cols.index("CO2_desert[ppm]") if "CO2_desert[ppm]" in numeric_cols else 0)
        if forecast_column:
            if time_cols:
                df_sorted = df.sort_values(by=time_cols[0])
            else:
                df_sorted = df.copy()

            if forecast_method == "ARIMA":
                st.subheader("ARIMA Forecast Settings")
                st.markdown("""
                **ARIMA Model (Default Order: (1,1,1))**  
                The ARIMA model uses past observations and forecast errors to predict future values.
                You only need to adjust the forecast horizon (number of future timesteps to predict).
                """)
                forecast_steps = st.number_input("Forecast Horizon (timesteps)", min_value=1, max_value=100, value=10)
                if st.button("Run ARIMA Forecast"):
                    series = df_sorted[forecast_column].dropna()
                    if len(series) < 10:
                        st.error("Not enough data points for ARIMA.")
                    else:
                        try:
                            forecast_vals, model_summary = forecast_arima(series, steps=forecast_steps)
                            st.write("**ARIMA Model Summary:**")
                            st.text(model_summary)
                            st.write("**Forecast Values:**")
                            st.write(forecast_vals)
                            # Interactive Plotly forecast chart with range slider and custom settings
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(series.index),
                                y=series,
                                mode='lines',
                                name='Historical'
                            ))
                            forecast_index = list(range(series.index[-1] + 1, series.index[-1] + forecast_steps + 1))
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_vals,
                                mode='lines+markers',
                                name='Forecast'
                            ))
                            fig.update_layout(
                                title=f"ARIMA Forecast for {forecast_column}",
                                xaxis_title="Time Index",
                                yaxis_title=forecast_column,
                                xaxis_rangeslider_visible=True,
                                hovermode='x'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"ARIMA forecasting failed: {e}")
            elif forecast_method == "Prophet" and prophet_available:
                st.subheader("Prophet Forecast Settings")
                st.markdown("""
                **Prophet Model**  
                Prophet automatically models trends and seasonality.
                Set the forecast horizon in days and the frequency of your data.
                """)
                forecast_steps = st.number_input("Forecast Horizon (days)", min_value=1, max_value=365, value=10)
                freq_choice = st.selectbox("Frequency", ["D", "H", "15min"], index=2)
                if time_cols:
                    date_col = time_cols[0]
                    if st.button("Run Prophet Forecast"):
                        df_prophet = df_sorted[[date_col, forecast_column]].dropna()
                        if len(df_prophet) < 10:
                            st.error("Not enough data points for Prophet.")
                        else:
                            try:
                                model, forecast_df = forecast_prophet(df_prophet, date_col, forecast_column, steps=forecast_steps, freq=freq_choice)
                                st.write("**Forecast DataFrame (last rows):**")
                                st.dataframe(forecast_df[['ds','yhat','yhat_lower','yhat_upper']].tail(20))
                                # Use Plotly for a custom interactive Prophet forecast plot
                                fig = px.line(forecast_df, x='ds', y='yhat', title=f"Prophet Forecast for {forecast_column}")
                                fig.update_layout(xaxis_title="Date", yaxis_title=forecast_column, xaxis_rangeslider_visible=True, hovermode='x')
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Prophet forecasting failed: {e}")
                else:
                    st.info("No valid time column found for Prophet forecasting.")

    ########################################
    # Tab: LLM Interpretation
    ########################################
    with tab_llm:
        st.header("LLM Interpretation of Forecast Results")
        default_summary = ("The forecast indicates a slight upward trend in the selected metric over the forecast period. "
                           "Please provide insights on what this trend might indicate for the ecosystem and suggest potential interventions.")
        forecast_summary = st.text_area("Enter a summary of forecast results for LLM interpretation", 
                                        value=default_summary, height=100)
        if st.button("Interpret Forecast with LLM"):
            prompt = f"""
            We have sensor data from Biosphere 2 with the following summary:
            {combined_summary[:1000]}  <!-- truncated for brevity -->

            Forecast Summary:
            {forecast_summary}

            Please provide actionable insights on how these forecast trends might affect the ecosystem, 
            and suggest potential interventions or further analyses.
            """
            try:
                llm_response = call_llm(llm_model1, prompt)
                st.subheader("LLM Forecast Interpretation")
                st.markdown(llm_response)
            except Exception as e:
                st.error(f"LLM call failed: {e}")

else:
    st.info("Please upload a CSV file or specify a folder path containing sensor data.")
