# B2Twin Hackathon Digital Twin Project

This project is a comprehensive dashboard for the B2Twin Hackathon at Biosphere 2. It integrates sensor data analysis with local LLM-powered insights to help monitor environmental parameters and provide actionable recommendations.

## Features

- **CSV File Upload:** Upload Biosphere 2 sensor data in CSV format.
- **Interactive Data Filtering:** Filter data by date range using an intuitive sidebar.
- **Visualizations:**
  - **💧 Water & Humidity Tracking:** Dual-axis time series plots, statistical summaries, and LLM-driven insights on water recycling efficiency.
  - **🔬 Rainforest CO₂ Monitoring:** Time series plots with anomaly detection, seasonal trend analysis, and detailed LLM insights.
- **LLM Integration:**
  - Generate detailed summaries of sensor data.
  - Simulate a conversation between two LLMs for deeper analysis and actionable recommendations.

## Project Structure

- **B2Twin_Project/**
  - **llm_utils.py:** Helper functions for LLM integration via Ollama (handles LLM calls and simulated conversations).
  - **streamlit_app.py:** Main Streamlit dashboard application for data upload, visualization, and LLM-powered analysis.
  - **requirements.txt:** Python dependencies for the project (e.g., streamlit, pandas, matplotlib).
  - **README.md:** Project documentation, setup instructions, and usage details.
  - **gitignore:** Gitignore file to exclude sensitive data.


## Setup Instructions

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/AI-Core-Biosphere-2/YourRepoName.git
    cd YourRepoName
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure Ollama is Installed:**
    - Download and install Ollama from [Ollama](https://ollama.com/).
    - Verify that your local LLM is working (for example, run `ollama run llama3.2`).

4. **Run the Streamlit App:**
    ```bash
    streamlit run streamlit_app.py
    ```

5. **Upload a CSV File:**
   - For **Water & Humidity Tracking**, ensure the CSV includes: `timestamp`, `water`, and `humidity` columns.
   - For **Rainforest CO₂ Monitoring**, ensure the CSV includes: `timestamp` and `co2` columns.

6. **Interact with the Dashboard:**
   - Use the sidebar to filter data by date and select your preferred LLM models.
   - Navigate between tabs to explore visualizations and trigger LLM-powered analyses.

## Customization

- **LLM Model Settings:** Adjust the model names in the sidebar (default is `llama3.2`) to use other models if desired.
- **Extend Analyses:** The code is modular—add new data analyses or visualizations by modifying `streamlit_app.py` and `llm_utils.py`.

## License

MIT License

## Contact

For any questions or support, please contact [Your Name/Team].