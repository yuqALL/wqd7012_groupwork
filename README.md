# River Water Quality Prediction System

A machine learning-based water quality analysis and prediction application using the CCME Water Quality Index (WQI).

## Project Structure

```
wqd7012_groupwork/
├── streamlit_app.py       # Main Streamlit application
├── run_streamlit.sh       # Script to run the application
├── requirements.txt       # Python dependencies
├── Combined_dataset.csv   # Preprocessed water quality dataset
├── rf_model.pkl          # Trained Random Forest model
├── xgb_model.pkl         # Trained XGBoost model
└── .gitattributes        # Git LFS configuration
```

## Features

- **Data Preprocessing**: Dataset overview, data filtering (2000-2023), and preprocessing pipeline
- **Exploratory Data Analysis (EDA)**: Distribution analysis, time series trends, and correlation heatmaps
- **Model Training**: Random Forest and XGBoost regression models for CCME WQI prediction
- **Interactive Prediction Tool**: Real-time water quality index prediction based on user inputs

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## Installation & Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yuqALL/wqd7012_groupwork.git
cd wqd7012_groupwork

# Run the setup script (first time only)
./run_streamlit.sh
```

The script will automatically:
1. Create a virtual environment (`streamlit_venv`)
2. Install all dependencies from `requirements.txt`
3. Launch the Streamlit application

### Manual Setup

```bash
# Create virtual environment
python3 -m venv streamlit_venv
source streamlit_venv/bin/activate  # Linux/Mac
# streamlit_venv\Scripts\activate   # Windows

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
- Country, Area, Waterbody Type
- Date
- Ammonia, Biochemical Oxygen Demand, Dissolved Oxygen
- Orthophosphate, pH, Temperature, Nitrogen, Nitrate
- CCME_Values (Target variable)

## Models

- **Random Forest Regressor**: Ensemble tree-based model for water quality prediction
- **XGBoost Regressor**: Gradient boosting model with optimized hyperparameters

Both models predict the CCME (Canadian Council of Ministers of the Environment) Water Quality Index.

## Navigation

Use the sidebar menu to navigate between sections:
1. Data Preprocessing (1.1, 1.2, 1.3)
2. Exploratory Data Analysis (2.1, 2.2, 2.3)
3. Model Training (3.1, 3.2, 3.3, 3.4)
4. Application Development (4.1)

## License

This project is for educational purposes as part of the WQD7012 course requirements.
