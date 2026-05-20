# 🏞️ River Water Quality Prediction and Contamination Detection System

**Live Application:** https://wqd7012groupwork-3swgv8agwcxjdihcyapopj.streamlit.app/

A machine learning-based water quality analysis and prediction application using the Water Quality Index (WQI).

## Features

- **Data Preprocessing**: Dataset overview, data filtering (2000-2023), and preprocessing pipeline
- **Exploratory Data Analysis (EDA)**: Distribution analysis, time series trends, and correlation heatmap
- **Model Training**: Random Forest, XGBoost and Hybrid NN-XGBoost regression models for Water Quality Index (WQI) prediction
- **Interactive Prediction Tool**: Real-time water quality index prediction based on user inputs

## Project Structure

```
wqd7012_groupwork/
├── data/
│   └── Combined_dataset.csv    # Preprocessed water quality dataset
│   └── River_Water_Quality.csv # Raw water quality dataset
│   └── River_Water_Quality_Final.csv # Processed water quality dataset
│   └── sample_input.csv # Sample dataset used to make prediction
│   └── demo_dashboard.csv # Sample dataset used to display dasboard
├── streamlit_app.py       # Main Streamlit application
├── run_streamlit.sh       # Script to run the application locally
├── requirements.txt       # Python dependencies
├── model/
│   └── rf_model.pkl           # Trained Random Forest model
│   └── xgb_model.pkl          # Trained XGBoost model
│   └── best_hybrid_nn_model.pth    # Trained Neural Network (NN) model
│   └── best_hybrid_nn_xgb_model.pth    # Trained Hybrid NN-XGBoost model
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── .gitattributes        # Git LFS configuration
```

## Local Installation & Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/24236510-ui/wqd7012_groupwork.git
cd wqd7012_groupwork

# Run the setup script (first time only)
./run_streamlit.sh
```

The script will automatically:
1. Create a virtual environment (`venv`)
2. Install all dependencies from `requirements.txt`
3. Launch the Streamlit application

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py --server.port 8501
```

### Access the Application

After launching, open your browser and navigate to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://0.0.0.0:8501

## Dataset

The `Combined_dataset.csv` contains water quality measurements from 2000 to 2023 with the following parameters:
- **Geographic**: Country, Area, Waterbody Type
- **Temporal**: Date
- **Water Quality Indicators**:
  - Ammonia (mg/l)
  - Biochemical Oxygen Demand (mg/l)
  - Dissolved Oxygen (mg/l)
  - Orthophosphate (mg/l)
  - pH (ph units)
  - Temperature (cel)
  - Nitrogen (mg/l)
  - Nitrate (mg/l)
- **Target Variable**: CCME_Values (Canadian Council of Ministers of the Environment Water Quality Index)

## Models

| Model | RMSE | MAE | R2 Score |
|-------|------|-----|----------|
| Random Forest | 1.2111 | 0.5897 | 0.9917 |
| XGBoost | 0.5190 | 0.2017 | 0.9985 |
| NN-XGBoost | 0.7126 | 0.2976 | 0.9971 |

The models predict the CCME Water Quality Index (WQI) based on water quality parameters.

## Navigation

Use the sidebar menu to navigate between sections:
1. **Application Development**

## License

This project is for educational purposes as part of the WQD7012 course requirements.
