import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from llm_utils import call_llm, simulate_llm_conversation
from data_pipeline import load_and_merge_csvs

def generate_combined_summary(df: pd.DataFrame) -> str:
    """
    Generate a combined summary of the uploaded DataFrame for both display and LLM analysis.
    Includes file shape, columns, data types, basic numeric statistics, and first 5 rows.
    """
    summary = f"**File Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n"
    summary += f"**Columns:** {', '.join(df.columns)}\n\n"
    summary += "**Data Types:**\n" + df.dtypes.to_frame(name="Data Type").to_markdown() + "\n\n"
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary += "**Statistical Summary for Numeric Columns:**\n" + df[numeric_cols].describe().to_markdown() + "\n\n"
    summary += "**First 5 Rows:**\n" + df.head().to_markdown() + "\n\n"
    return summary

def display_llm_output(llm_output: str):
    """
    Display LLM output nicely.
    If the output contains bullet points or list formatting, use st.markdown; otherwise, st.write.
    """
    if any(line.strip().startswith(('-', '*', '1.')) for line in llm_output.splitlines()):
        st.markdown(llm_output)
    else:
        st.write(llm_output)

def get_time_column(df: pd.DataFrame) -> str:
    """
    Try to detect a time column by checking for columns that contain 'time' in their name (case insensitive).
    Returns the column name if found, otherwise None.
    """
    for col in df.columns:
        if 'time' in col.lower():
            return col
    return None

# Set up the main page
st.set_page_config(page_title="B2Twin Hackathon: Biosphere 2 Digital Twin Dashboard", layout="wide")
st.title("B2Twin Hackathon: Biosphere 2 Digital Twin Dashboard")
st.markdown("""
This advanced dashboard integrates sensor data analysis with local LLM-powered insights.
It dynamically analyzes any uploaded sensor data—regardless of its format—providing actionable insights to help scientists restore degraded environments on Earth and prepare for space travel.
""")

# Sidebar for file upload and settings
st.sidebar.header("Upload and Settings")
data_input_method = st.sidebar.radio("Data Input Method", ("Single CSV File", "Folder of CSV Files"))

# Dropdown for switching between available LLM models
llm_options = ["llama3.2", "mistral", "llama2", "phi"]
llm_model1 = st.sidebar.selectbox("Primary LLM Model", options=llm_options, index=0)
llm_model2 = st.sidebar.selectbox("Secondary LLM Model", options=llm_options, index=0)

# Data ingestion logic
df = None
if data_input_method == "Single CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Try to handle a timestamp column
            if 'timestamp' in df.columns or 'DateTime' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'DateTime'
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
elif data_input_method == "Folder of CSV Files":
    folder_path = st.sidebar.text_input("Enter folder path for CSV files")
    if folder_path:
        try:
            df = load_and_merge_csvs(folder_path)
            # Try to handle a timestamp column
            if 'timestamp' in df.columns or 'DateTime' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'DateTime'
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])
        except Exception as e:
            st.error(f"Error loading folder: {e}")

if df is not None:
    st.subheader("Uploaded File Summary")
    combined_summary = generate_combined_summary(df)
    st.markdown(combined_summary)
    
    # Create three tabs: 
    # 1) General LLM Analysis 
    # 2) Data Visualization 
    # 3) Advanced Analytics
    tab1, tab2, tab3 = st.tabs(["General LLM Analysis", "Data Visualization", "Advanced Analytics"])
    
    # Tab 1: General LLM Analysis
    with tab1:
        st.header("General LLM Analysis")
        if st.button("🧠 Analyze Data with LLM (Generic)"):
            prompt = (
                "Analyze the following Biosphere 2 sensor data and provide insights about the "
                "ecosystem's health and potential areas for intervention:\n\n"
                f"{combined_summary}"
            )
            llm_response = call_llm(llm_model1, prompt)
            st.subheader("LLM Analysis")
            display_llm_output(llm_response)
        
        st.markdown("---")
        st.subheader("LLM Conversation Simulation")
        if st.button("🤖 Simulate LLM Conversation"):
            resp1, resp2 = simulate_llm_conversation(combined_summary, llm_model1, llm_model2)
            st.markdown("**Model 1 Response**")
            display_llm_output(resp1)
            st.markdown("**Model 2 Response**")
            display_llm_output(resp2)

    # Tab 2: Data Visualization
    with tab2:
        st.header("Data Visualization Dashboard")
        time_col = get_time_column(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            chart_type = st.selectbox("Select Chart Type", ["Line Plot", "Scatter Plot", "Histogram", "Box Plot"])
            
            if time_col and chart_type in ["Line Plot", "Scatter Plot"]:
                # For time-based plots
                col_choice = st.selectbox("Select a numeric column to plot over time", numeric_cols)
                if col_choice:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if chart_type == "Line Plot":
                        ax.plot(df[time_col], df[col_choice], marker='o', linestyle='-')
                    else:  # Scatter Plot
                        ax.scatter(df[time_col], df[col_choice])
                    ax.set_xlabel("Time")
                    ax.set_ylabel(col_choice)
                    ax.set_title(f"{col_choice} over Time ({chart_type})")
                    st.pyplot(fig)
            else:
                # For non-time or other chart types
                col_choice = st.selectbox("Select a numeric column", numeric_cols)
                if col_choice:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if chart_type == "Histogram":
                        ax.hist(df[col_choice].dropna(), bins=20)
                        ax.set_xlabel(col_choice)
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Distribution of {col_choice}")
                    elif chart_type == "Box Plot":
                        ax.boxplot(df[col_choice].dropna())
                        ax.set_ylabel(col_choice)
                        ax.set_title(f"Box Plot of {col_choice}")
                    st.pyplot(fig)
        else:
            st.info("No numeric columns detected in your dataset to visualize.")

    # Tab 3: Advanced Analytics
    with tab3:
        st.header("Advanced Analytics")
        st.markdown("Here you can generate more in-depth analyses like correlation heatmaps.")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            st.info("Not enough numeric columns to compute correlations.")
        else:
            if st.button("Show Correlation Heatmap"):
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
else:
    st.info("Please upload a CSV file or specify a folder path containing CSV files for Biosphere 2 sensor data.")
