# River Water Quality Prediction System

**Live Application:** https://wqd7012groupwork-3swgv8agwcxjdihcyapopj.streamlit.app/

A machine learning-based water quality analysis and prediction application using the CCME Water Quality Index (WQI).

## Features

- **Data Preprocessing**: Dataset overview, data filtering (2000-2023), and preprocessing pipeline
- **Exploratory Data Analysis (EDA)**: Distribution analysis, time series trends, and correlation heatmaps
- **Model Training**: Random Forest, XGBoost, Hybrid CNN, and Hybrid CNN-XGBoost regression models for CCME WQI prediction
- **Interactive Prediction Tool**: Real-time water quality index prediction based on user inputs

## Project Structure

```
wqd7012_groupwork/
├── streamlit_app.py       # Main Streamlit application
├── run_streamlit.sh       # Script to run the application locally
├── requirements.txt       # Python dependencies
├── train_models.py        # Script to train and save all models
├── rf_model.pkl          # Trained Random Forest model
├── xgb_model.pkl         # Trained XGBoost model
├── best_hybrid_cnn.pth   # Trained Hybrid CNN model (PyTorch)
├── hybrid_xgb_model.pkl  # Trained Hybrid CNN-XGBoost model
├── images/
│   ├── rf_evaluation.png       # Random Forest evaluation visualization
│   ├── xgb_evaluation.png     # XGBoost evaluation visualization
│   ├── hybrid_evaluation.png   # Hybrid CNN-XGBoost evaluation visualization
│   ├── model_comparison.png   # Model comparison bar charts
│   ├── eda_distribution.png   # EDA distribution plots
│   ├── eda_timeseries.png    # EDA time series plots
│   └── eda_correlation.png   # EDA correlation heatmap
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

### Training Models

To train all models from scratch:

```bash
python train_models.py
```

This will generate:
- `rf_model.pkl` - Random Forest model
- `xgb_model.pkl` - XGBoost model
- `best_hybrid_cnn.pth` - Hybrid CNN model (PyTorch)
- `hybrid_xgb_model.pkl` - Hybrid CNN-XGBoost model

### Access the Application

After launching, open your browser and navigate to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://0.0.0.0:8501

## Dataset

The water quality dataset contains measurements from 2000 to 2023 with the following parameters:
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

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | 0.1507 | 0.0127 | 0.9999 |
| XGBoost | 0.1311 | 0.0109 | 0.9999 |
| Hybrid CNN | 0.3500 | 0.2500 | 0.9800 |
| Hybrid CNN-XGBoost | 5.3968 | 3.1332 | 0.5559 |

### Model Descriptions

- **Random Forest**: Ensemble of decision trees for robust regression
- **XGBoost**: Gradient boosting with high predictive accuracy
- **Hybrid CNN**: 1D Convolutional Neural Network with batch normalization for feature extraction
- **Hybrid CNN-XGBoost**: Combines CNN feature extraction with XGBoost regression for time-series prediction

## Navigation

Use the sidebar menu to navigate between sections:
1. **Data Preprocessing** (1.1, 1.2, 1.3)
2. **Exploratory Data Analysis** (2.1, 2.2, 2.3)
3. **Model Training** (3.1, 3.2, 3.3, 3.4, 3.5)
4. **Application Development** (4.1)