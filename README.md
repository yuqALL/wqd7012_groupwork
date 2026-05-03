# River Water Quality Prediction System

A machine learning-based water quality analysis and prediction application using the CCME Water Quality Index (WQI).

## Features

- **Data Preprocessing**: Dataset overview, data filtering (2000-2023), and preprocessing pipeline
- **Exploratory Data Analysis (EDA)**: Distribution analysis, time series trends, and correlation heatmaps
- **Model Training**: Random Forest and XGBoost regression models for CCME WQI prediction
- **Interactive Prediction Tool**: Real-time water quality index prediction based on user inputs

## Project Structure

```
wqd7012_groupwork/
├── streamlit_app.py       # Main Streamlit application
├── run_streamlit.sh       # Script to run the application locally
├── requirements.txt       # Python dependencies
├── Combined_dataset.csv   # Preprocessed water quality dataset (managed by Git LFS)
├── River_Water_Quality.csv # Raw water quality dataset (managed by Git LFS)
├── rf_model.pkl          # Trained Random Forest model
├── xgb_model.pkl         # Trained XGBoost model
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

## Deployment to Streamlit Community Cloud

### Prerequisites

1. Push all code to your GitHub repository
2. Ensure Git LFS is properly configured for large files (.csv, .pkl)

### Steps

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **New app**
3. Select your GitHub repository: `24236510-ui/wqd7012_groupwork`
4. Select branch: `main`
5. Main file path: `streamlit_app.py`
6. Click **Deploy**

### Note on Large Files

This application uses Git LFS (Large File Storage) to manage large data files. The data files will be downloaded automatically when the app starts.

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

- **Random Forest Regressor**: Ensemble tree-based model for water quality prediction
- **XGBoost Regressor**: Gradient boosting model with optimized hyperparameters

Both models predict the CCME Water Quality Index (WQI) based on water quality parameters.

## Navigation

Use the sidebar menu to navigate between sections:
1. **Data Preprocessing** (1.1, 1.2, 1.3)
2. **Exploratory Data Analysis** (2.1, 2.2, 2.3)
3. **Model Training** (3.1, 3.2, 3.3, 3.4)
4. **Application Development** (4.1)

## Troubleshooting

### Memory Issues on Streamlit Cloud

If you encounter memory errors during deployment:
- The app is optimized to use sampling for large datasets
- Data files are downloaded from GitHub on startup
- Models are trained on sampled data to reduce memory usage

### Version Compatibility

- Python 3.9 - 3.12 recommended
- See `requirements.txt` for exact package versions

## License

This project is for educational purposes as part of the WQD7012 course requirements.
