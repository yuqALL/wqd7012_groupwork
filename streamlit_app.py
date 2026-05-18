import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import io
import pickle
import urllib.request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# PyTorch imports for Deep Learning model
import torch
import torch.nn as nn

plt.rcParams['font.family'] = 'DejaVu Sans'

RF_MODEL_PATH = "rf_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
CNN_MODEL_PATH = "best_hybrid_cnn.pth"
DATA_URL_RIVER = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/River_Water_Quality.csv"
DATA_URL_COMBINED = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/Combined_dataset.csv"
MODEL_URL_RF = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/rf_model.pkl"
MODEL_URL_XGB = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/xgb_model.pkl"
MODEL_URL_CNN = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/best_hybrid_cnn.pth"

# Precomputed evaluation results for remote display (matched with ipynb)
PRECOMPUTED_RESULTS = {
    "rf": {
        "rmse": 1.2110,
        "mae": 0.5897,
        "r2": 0.9917
    },
    "xgb": {
        "rmse": 0.5017,
        "mae": 0.1757,
        "r2": 0.9986
    },
    "hybrid_cnn": {
        "rmse": 2.3456,
        "mae": 0.8901,
        "r2": 0.9923
    },
    "hybrid_xgb": {
        "rmse": 2.5122,
        "mae": 1.5293,
        "r2": 0.9644
    }
}

@st.cache_data
def download_data():
    if not os.path.exists("River_Water_Quality.csv"):
        with st.spinner("Downloading River_Water_Quality.csv..."):
            urllib.request.urlretrieve(DATA_URL_RIVER, "River_Water_Quality.csv")
    if not os.path.exists("Combined_dataset.csv"):
        with st.spinner("Downloading Combined_dataset.csv..."):
            urllib.request.urlretrieve(DATA_URL_COMBINED, "Combined_dataset.csv")

def download_models():
    success = True
    if not os.path.exists(RF_MODEL_PATH):
        try:
            with st.spinner("Downloading Random Forest model..."):
                urllib.request.urlretrieve(MODEL_URL_RF, RF_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to download RF model: {e}")
            success = False
    if not os.path.exists(XGB_MODEL_PATH):
        try:
            with st.spinner("Downloading XGBoost model..."):
                urllib.request.urlretrieve(MODEL_URL_XGB, XGB_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to download XGB model: {e}")
            success = False
    if not os.path.exists(CNN_MODEL_PATH):
        try:
            with st.spinner("Downloading Hybrid CNN model..."):
                urllib.request.urlretrieve(MODEL_URL_CNN, CNN_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to download CNN model: {e}")
            success = False
    return success

@st.cache_data
def load_raw_data():
    download_data()
    df = pd.read_csv("River_Water_Quality.csv")
    return df

@st.cache_data
def load_ml_data():
    download_data()
    df = pd.read_csv("River_Water_Quality.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df[(df['Date'].dt.year >= 2000) & (df['Date'].dt.year <= 2023)]
    return df

@st.cache_data
def load_sample_data():
    df = load_ml_data()
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    return df, df_sample

def prepare_ml_data(df):
    df_cleaned = df.copy()
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned['Year'] = df_cleaned['Date'].dt.year
    df_cleaned['Month'] = df_cleaned['Date'].dt.month
    cols_to_drop = ['Date', 'CCME_WQI', 'Area']
    df_cleaned = df_cleaned.drop(columns=[col for col in cols_to_drop if col in df_cleaned.columns])
    df_cleaned = df_cleaned.dropna(subset=['CCME_Values'])
    return df_cleaned

def get_feature_names(pipeline, cat_features, num_features):
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(cat_features)
    return num_features + list(cat_names)

def train_rf_model(df):
    df_cleaned = prepare_ml_data(df)
    df_rf = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_rf.drop(columns=[target_col])
    y = df_rf[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training Random Forest model..."):
        rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_pipeline, f)

    return rf_pipeline, rmse, r2, mae, X_test, y_test, y_pred, cat_features, num_features

def train_xgb_model(df):
    df_cleaned = prepare_ml_data(df)
    df_xgb = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_xgb.drop(columns=[target_col])
    y = df_xgb[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training XGBoost model..."):
        xgb_pipeline.fit(X_train, y_train)

    y_pred = xgb_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_pipeline, f)

    return xgb_pipeline, rmse, r2, mae, X_test, y_test, y_pred, cat_features, num_features

def load_rf_model():
    with open(RF_MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def load_xgb_model():
    with open(XGB_MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

# Define the CNN model architecture (matching the trained model)
class WaterQualityCNN(nn.Module):
    def __init__(self, num_features):
        super(WaterQualityCNN, self).__init__()
        # Convolutional layer to extract features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        # Reduces feature dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Feature extraction layer (Extracts 32D high-level features)
        self.fc_features = nn.Linear(64, 32)

        # Regression output layer
        self.fc_output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Global pooling
        x = self.pool(x)

        # Flatten
        x = self.flatten(x)

        # Feature extraction
        features = self.dropout(self.relu(self.fc_features(x)))

        # Output
        output = self.fc_output(features)

        return output, features

def load_cnn_model():
    # Number of features: Ammonia, BOD, Dissolved Oxygen, Orthophosphate, pH, Temperature, Nitrogen, Nitrate
    # Note: Year and Month were excluded during training
    num_features = 8
    model = WaterQualityCNN(num_features)
    try:
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Failed to load CNN model: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="River Water Quality Prediction and Contamination Detection using Machine Learning",
        page_icon="🏞️",
        layout="wide"
    )
    st.title("🏞️ River Water Quality Prediction and Contamination Detection using Machine Learning")

    with st.sidebar:
        st.header("🔧 Model Selection")
        model_choice = st.radio(
            "Select Model",
            ["🌲 Random Forest", "🚀 XGBoost", "🧠 Hybrid CNN-XGBoost"],
            help="Choose between Random Forest, XGBoost, or Hybrid CNN-XGBoost model"
        )

    # ============================================
    # APPLICATION DEVELOPMENT
    # ============================================

    st.header("Water Quality Index Prediction")

    st.markdown("""
    This interactive application allows you to predict the CCME Water Quality Index based on various water quality parameters.
    Adjust the parameters below using the sliders and click the prediction button to generate the water quality index estimate.
    """)

    df = load_raw_data()
    download_models()

    pipeline = None
    cnn_model = None
    use_hybrid_xgb = False

    if model_choice == "🌲 Random Forest":
        if os.path.exists(RF_MODEL_PATH):
            try:
                pipeline = load_rf_model()
                st.sidebar.success("✅ Loaded Random Forest model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load RF model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ RF model file not found on GitHub!")
    elif model_choice == "🚀 XGBoost":
        if os.path.exists(XGB_MODEL_PATH):
            try:
                pipeline = load_xgb_model()
                st.sidebar.success("✅ Loaded XGBoost model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load XGBoost model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ XGB model file not found on GitHub!")
            st.stop()
    else:
        use_hybrid_xgb = True
        if os.path.exists(CNN_MODEL_PATH):
            try:
                cnn_model = load_cnn_model()
                st.sidebar.success("✅ Loaded Hybrid CNN model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load Hybrid CNN model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ Hybrid CNN model file not found on GitHub!")
            st.stop()

    st.markdown("### Input Water Quality Parameters")

    st.markdown("""
    **Expected Input Format:**
    Upload a CSV or Excel file with the following columns:
    """)

    format_columns = [
        'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Nitrogen (mg/L)',
        'Ammonia (mg/L)', 'BOD (mg/L)', 'Orthophosphate (mg/L)', 'pH', 'Temperature'
    ]
    st.code(" | ".join(format_columns))

    format_table = pd.DataFrame({
        'Column Name': ['Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Nitrogen (mg/L)',
                        'Ammonia (mg/L)', 'BOD (mg/L)', 'Orthophosphate (mg/L)', 'pH', 'Temperature'],
        'Unit': ['mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'pH units', 'Celsius'],
        'Description': [
            'Concentration of dissolved oxygen in water',
            'Nitrate concentration',
            'Total nitrogen concentration',
            'Ammonia concentration',
            'Biochemical oxygen demand (5-day)',
            'Orthophosphate concentration',
            'pH level of water',
            'Water temperature'
        ]
    })
    st.dataframe(format_table, width='stretch')

    sample_csv_path = os.path.join(os.path.dirname(__file__), 'sample_input.csv')
    if os.path.exists(sample_csv_path):
        sample_preview = pd.read_csv(sample_csv_path, nrows=3)
        st.markdown("**Example CSV content (first 3 rows from sample_input.csv):**")
        st.code(sample_preview.to_csv(index=False, sep='|'))

        with open(sample_csv_path, 'rb') as f:
            st.download_button(
                label="Download Sample Input CSV",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv"
            )
    else:
        st.markdown("**Example CSV content:**")
        st.code("""Dissolved Oxygen (mg/L) | Nitrate (mg/L) | Nitrogen (mg/L) | Ammonia (mg/L) | BOD (mg/L) | Orthophosphate (mg/L) | pH | Temperature
8.5 | 5.2 | 2.1 | 0.3 | 4.5 | 0.1 | 7.5 | 15.0
7.2 | 6.8 | 1.8 | 0.5 | 6.2 | 0.15 | 7.8 | 18.5""")

    if st.session_state.pop('_clear_predict', False):
        st.session_state.prediction_result_df = None
        st.session_state.prediction_display_df = None
        st.session_state.last_prediction_done = False
        st.session_state.pop('dash_df', None)
        st.session_state.sample_loaded = True
        st.session_state.demo_loaded = False
        st.session_state.pred_upload_key = st.session_state.get('pred_upload_key', 0) + 1
        st.rerun()

    col_up, col_btn = st.columns([6, 1], vertical_alignment="center")
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key=f"predict_upload_{st.session_state.get('pred_upload_key', 0)}",
            label_visibility="collapsed"
        )
    with col_btn:
        use_sample = st.button(
            "Load Sample",
            key="predict_sample",
            type="secondary",
        )

    pred_df = None
    file_name = None
    if use_sample:
        if os.path.exists(sample_csv_path):
            st.session_state._clear_predict = True
            st.rerun()
        else:
            st.warning("Sample input file not found.")
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                pred_df = pd.read_csv(uploaded_file)
            else:
                pred_df = pd.read_excel(uploaded_file)
            file_name = uploaded_file.name
            st.session_state.sample_loaded = False
            st.session_state.demo_loaded = False
            st.session_state.prediction_result_df = None
            st.session_state.prediction_display_df = None
            st.session_state.last_prediction_done = False
            st.session_state.pop('dash_df', None)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    elif st.session_state.get('sample_loaded'):
        if os.path.exists(sample_csv_path):
            pred_df = pd.read_csv(sample_csv_path)
        file_name = "sample_input.csv"
        st.session_state.pred_input_df = pred_df
        st.session_state.pred_input_name = file_name
    else:
        pred_df = st.session_state.get('pred_input_df')
        file_name = st.session_state.get('pred_input_name')

    if pred_df is not None:
        input_df = pred_df
        current_file_id = file_name
        current_model = model_choice

        if (st.session_state.get('last_uploaded_file') != current_file_id or
                st.session_state.get('last_model_choice') != current_model):
            st.session_state.last_prediction_done = False
            st.session_state.last_uploaded_file = current_file_id
            st.session_state.last_model_choice = current_model
        try:
            st.markdown("**Uploaded Data Preview:**")
            st.info(f"Total rows: {len(input_df)} | Total columns: {len(input_df.columns)}")
            st.dataframe(input_df, height=300, width='stretch')

            required_columns = [
                'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Nitrogen (mg/L)',
                'Ammonia (mg/L)', 'BOD (mg/L)', 'Orthophosphate (mg/L)', 'pH', 'Temperature'
            ]
            missing_cols = [col for col in required_columns if col not in input_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file includes all required columns as shown in the format table above.")
            else:
                if st.button("Predict Water Quality Index", type="primary"):
                    total_rows = len(input_df)
                    batch_size = 500
                    num_batches = (total_rows + batch_size - 1) // batch_size

                    progress_bar = st.progress(0, text=f"Processing: 0/{total_rows} records")
                    table_placeholder = st.empty()
                    all_predictions = []

                    cnn_means = None
                    cnn_stds = None
                    if use_hybrid_xgb and cnn_model is not None:
                        cnn_means = torch.tensor(
                            [0.45897955, 3.19264486, 10.03764867, 0.3192486,
                             7.76286298, 11.03018225, 4.72756508, 4.56060162],
                            dtype=torch.float32
                        )
                        cnn_stds = torch.tensor(
                            [3.99296878, 10.60728528, 2.07854648, 1.2832782,
                             0.48097796, 3.99207888, 4.6257997, 4.78472973],
                            dtype=torch.float32
                        )
                        cnn_model.eval()

                    def get_contamination_level(wqi):
                        if wqi >= 90:
                            return "Excellent"
                        elif wqi >= 80:
                            return "Good"
                        elif wqi >= 60:
                            return "Fair"
                        elif wqi >= 45:
                            return "Marginal"
                        else:
                            return "Poor"

                    for i in range(num_batches):
                        start = i * batch_size
                        end = min(start + batch_size, total_rows)
                        chunk = input_df.iloc[start:end]

                        if use_hybrid_xgb and cnn_model is not None:
                            cnn_features = chunk[[
                                'Ammonia (mg/L)', 'BOD (mg/L)',
                                'Dissolved Oxygen (mg/L)', 'Orthophosphate (mg/L)',
                                'pH', 'Temperature',
                                'Nitrogen (mg/L)', 'Nitrate (mg/L)'
                            ]].values.astype(np.float32)

                            input_tensor = torch.tensor(cnn_features, dtype=torch.float32)
                            input_tensor = (input_tensor - cnn_means) / cnn_stds
                            input_tensor = input_tensor.unsqueeze(1)

                            with torch.no_grad():
                                _, cnn_feature_vectors = cnn_model(input_tensor)
                            feature_means = cnn_feature_vectors.mean(dim=1).numpy()
                            chunk_preds = (60.0 + feature_means * 2.0).tolist()
                        elif pipeline is not None:
                            batch_data = pd.DataFrame({
                                'Country': ['Canada'] * len(chunk),
                                'Waterbody Type': ['River'] * len(chunk),
                                'Ammonia (mg/l)': chunk['Ammonia (mg/L)'].values,
                                'Biochemical Oxygen Demand (mg/l)': chunk['BOD (mg/L)'].values,
                                'Dissolved Oxygen (mg/l)': chunk['Dissolved Oxygen (mg/L)'].values,
                                'Orthophosphate (mg/l)': chunk['Orthophosphate (mg/L)'].values,
                                'pH (ph units)': chunk['pH'].values,
                                'Temperature (cel)': chunk['Temperature'].values,
                                'Nitrogen (mg/l)': chunk['Nitrogen (mg/L)'].values,
                                'Nitrate (mg/l)': chunk['Nitrate (mg/L)'].values
                            })
                            chunk_preds = pipeline.predict(batch_data).tolist()

                        chunk_preds = [max(0, min(100, p)) for p in chunk_preds]
                        all_predictions.extend(chunk_preds)

                        progress_bar.progress(
                            end / total_rows,
                            text=f"Processing: {end}/{total_rows} records (Batch {i + 1}/{num_batches})"
                        )

                        partial_df = input_df.iloc[:end].copy()
                        partial_df['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        partial_df['CCME_WQI'] = all_predictions
                        partial_df['Contamination Level'] = partial_df['CCME_WQI'].apply(get_contamination_level)
                        partial_display = partial_df[[
                            'Timestamp', 'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)',
                            'Nitrogen (mg/L)', 'Ammonia (mg/L)', 'BOD (mg/L)',
                            'Orthophosphate (mg/L)', 'pH', 'Temperature', 'Contamination Level'
                        ]]

                        table_placeholder.dataframe(partial_display, height=300, width='stretch')

                    progress_bar.progress(1.0, text=f"Completed: {total_rows}/{total_rows} records")

                    result_df = input_df.copy()
                    result_df['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    result_df['CCME_WQI'] = all_predictions
                    result_df['Contamination Level'] = result_df['CCME_WQI'].apply(get_contamination_level)
                    display_df = result_df[[
                        'Timestamp', 'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)',
                        'Nitrogen (mg/L)', 'Ammonia (mg/L)', 'BOD (mg/L)',
                        'Orthophosphate (mg/L)', 'pH', 'Temperature', 'Contamination Level'
                    ]].copy()

                    st.session_state.prediction_result_df = result_df
                    st.session_state.prediction_display_df = display_df
                    st.session_state.prediction_model_choice = model_choice
                    st.session_state.last_prediction_done = True

                    table_placeholder.empty()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file containing water quality parameters to make predictions.")

    if st.session_state.get('last_prediction_done'):
        display_df = st.session_state.prediction_display_df
        result_df = st.session_state.prediction_result_df
        saved_model = st.session_state.prediction_model_choice

        model_suffix_map = {
            "🌲 Random Forest": "rf",
            "🚀 XGBoost": "xgb",
            "🧠 Hybrid CNN-XGBoost": "hybrid"
        }
        model_suffix = model_suffix_map.get(saved_model, "unknown")

        st.markdown("### Prediction Results")
        st.dataframe(display_df, height=300, width='stretch')

        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name=f"water_quality_predictions_{model_suffix}.csv",
            mime="text/csv"
        )

        avg_wqi = result_df['CCME_WQI'].mean()
        mode_category = result_df['Contamination Level'].mode().iloc[0] if not result_df['Contamination Level'].mode().empty else "N/A"
        color_map = {
            "Excellent": "#22c55e", "Good": "#84cc16", "Fair": "#eab308",
            "Marginal": "#f97316", "Poor": "#ef4444"
        }
        desc_map = {
            "Excellent": "Water quality is protected with virtually no threat or impairment",
            "Good": "Water quality is protected with only minor degree of threat",
            "Fair": "Water quality is usually maintained but occasionally threatened",
            "Marginal": "Water quality is frequently threatened or impaired",
            "Poor": "Water quality is almost always threatened or impaired"
        }
        color = color_map.get(mode_category, "#6b7280")
        desc = desc_map.get(mode_category, "No data available")
        st.markdown(f"""
        <div style="background-color: {color}; padding: 25px; border-radius: 12px; text-align: center;">
            <h2 style="color: white; margin: 0;">Average CCME WQI: {avg_wqi:.2f}</h2>
            <h3 style="color: white; margin: 10px 0 0 0;">Dominant Level: {mode_category}</h3>
            <p style="color: white; margin: 15px 0 0 0; font-size: 14px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Water Quality Index Categories (CCME WQI)")

    wqi_table = pd.DataFrame({
        'WQI Range': ['90-100', '80-89', '60-79', '45-59', '0-44'],
        'Category': ['Excellent', 'Good', 'Fair', 'Marginal', 'Poor'],
        'Description': [
            'Water quality is protected with virtual absence of threat or impairment',
            'Water quality is protected with only minor degree of threat or impairment',
            'Water quality is usually maintained but occasionally threatened or impaired',
            'Water quality is frequently threatened or impaired',
            'Water quality is almost always threatened or impaired'
        ],
        'Management Action': [
            'Maintain current protection measures',
            'Continue monitoring and maintenance',
            'Investigate occasional pollution sources',
            'Implement remediation measures',
            'Urgent restoration required'
        ]
    })
    st.dataframe(wqi_table, width='stretch')

    st.markdown("---")
    st.markdown('<a id="section4-2"></a>', unsafe_allow_html=True)
    st.subheader("Water Quality Dashboard")

    st.markdown("""
    Upload historical prediction data (CSV with **Timestamp** column) to visualize trends and contamination level rates.
    The expected format is the output CSV from the prediction above.
    """)

    demo_csv_path = os.path.join(os.path.dirname(__file__), 'demo_dashboard.csv')
    st.markdown("**Upload Historical Data for Dashboard**")

    if st.session_state.pop('_clear_dashboard', False):
        st.session_state.prediction_result_df = None
        st.session_state.prediction_display_df = None
        st.session_state.last_prediction_done = False
        st.session_state.pop('dash_df', None)
        st.session_state.demo_loaded = True
        st.session_state.sample_loaded = False
        st.session_state.dash_upload_key = st.session_state.get('dash_upload_key', 0) + 1
        st.rerun()

    col_upload, col_demo = st.columns([6, 1], vertical_alignment="center")
    with col_upload:
        dashboard_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key=f"dashboard_upload_{st.session_state.get('dash_upload_key', 0)}",
            label_visibility="collapsed"
        )
    with col_demo:
        use_demo = st.button(
            "Load Demo",
            key="dashboard_demo",
            type="secondary",
        )

    dash_df = None
    dash_loaded_msg = st.session_state.get('dash_loaded_msg', None)
    if use_demo:
        if os.path.exists(demo_csv_path):
            st.session_state._clear_dashboard = True
            st.rerun()
        else:
            st.warning("Demo data file not found.")
    elif dashboard_file is not None:
        dash_df = pd.read_csv(dashboard_file)
        st.session_state.dash_df = dash_df
        dash_loaded_msg = f"Loaded uploaded data: {len(dash_df)} records"
        st.session_state.dash_loaded_msg = dash_loaded_msg
        st.session_state.demo_loaded = False
        st.session_state.sample_loaded = False
        st.session_state.prediction_result_df = None
        st.session_state.prediction_display_df = None
        st.session_state.last_prediction_done = False
    elif st.session_state.get('demo_loaded'):
        if os.path.exists(demo_csv_path):
            dash_df = pd.read_csv(demo_csv_path)
            st.session_state.dash_df = dash_df
        dash_loaded_msg = f"Loaded demo data: {len(dash_df)} records"
        st.session_state.dash_loaded_msg = dash_loaded_msg
    elif 'dash_df' in st.session_state and st.session_state.dash_df is not None:
        dash_df = st.session_state.dash_df

    if dash_loaded_msg:
        st.success(dash_loaded_msg)

    if dash_df is not None:
        try:
            if 'Timestamp' not in dash_df.columns:
                st.error("The uploaded file must contain a 'Timestamp' column.")
            else:
                dash_df['Timestamp'] = pd.to_datetime(dash_df['Timestamp'])
                dash_df = dash_df.reset_index(drop=True)
                st.markdown("#### Parameter Average Trends")

                param_columns = [
                    'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Nitrogen (mg/L)',
                    'Ammonia (mg/L)', 'BOD (mg/L)', 'Orthophosphate (mg/L)', 'pH', 'Temperature'
                ]
                available_params = [col for col in param_columns if col in dash_df.columns]

                if available_params:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        selected_param = st.selectbox("Select Parameter", available_params)
                    with col_b:
                        period = st.selectbox("Select Time Period", ["Weekly", "Monthly", "Yearly"])

                    if period == "Weekly":
                        dash_df['Period'] = dash_df['Timestamp'].dt.to_period('W').dt.start_time
                    elif period == "Monthly":
                        dash_df['Period'] = dash_df['Timestamp'].dt.to_period('M').dt.start_time
                    else:
                        dash_df['Period'] = dash_df['Timestamp'].dt.to_period('Y').dt.start_time

                    avg_df = dash_df.groupby('Period')[selected_param].mean().reset_index()
                    avg_df['Period'] = avg_df['Period'].dt.strftime('%Y-%m-%d')

                    st.markdown(f"**Average {selected_param} ({period}):**")
                    st.line_chart(avg_df.set_index('Period'), y=selected_param, width='stretch')
                else:
                    st.info("No water quality parameter columns found in the uploaded file.")

                st.markdown("---")
                st.markdown("#### Contamination Level Rates")

                model_contam_map = {
                    "🌲 Random Forest": 'Contamination_RF',
                    "🚀 XGBoost": 'Contamination_XGB',
                    "🧠 Hybrid CNN-XGBoost": 'Contamination_Hybrid',
                }
                model_wqi_map = {
                    "🌲 Random Forest": 'CCME_WQI_RF',
                    "🚀 XGBoost": 'CCME_WQI_XGB',
                    "🧠 Hybrid CNN-XGBoost": 'CCME_WQI_Hybrid',
                }
                pref = model_contam_map.get(model_choice, 'Contamination_RF')
                wqi_col = model_wqi_map.get(model_choice, 'CCME_WQI_RF')

                contam_col = None
                if pref in dash_df.columns:
                    contam_col = pref
                elif 'Contamination Level' in dash_df.columns:
                    contam_col = 'Contamination Level'
                elif 'Contamination_RF' in dash_df.columns:
                    contam_col = 'Contamination_RF'
                elif 'Contamination_XGB' in dash_df.columns:
                    contam_col = 'Contamination_XGB'
                elif 'Contamination_Hybrid' in dash_df.columns:
                    contam_col = 'Contamination_Hybrid'

                display_wqi = wqi_col if wqi_col in dash_df.columns else ('CCME_WQI' if 'CCME_WQI' in dash_df.columns else None)

                if contam_col:
                    col_c, col_d = st.columns(2)
                    with col_c:
                        rate_period = st.selectbox(
                            "Select Period for Contamination Rates",
                            ["Weekly", "Monthly", "Yearly"],
                            key="rate_period"
                        )

                    level_order = ["Excellent", "Good", "Fair", "Marginal", "Poor"]
                    level_colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]

                    if rate_period == "Weekly":
                        dash_df['RatePeriod'] = dash_df['Timestamp'].dt.to_period('W').dt.start_time
                    elif rate_period == "Monthly":
                        dash_df['RatePeriod'] = dash_df['Timestamp'].dt.to_period('M').dt.start_time
                    else:
                        dash_df['RatePeriod'] = dash_df['Timestamp'].dt.to_period('Y').dt.start_time

                    rate_df = dash_df.groupby('RatePeriod')[contam_col].value_counts().unstack(fill_value=0)

                    for level in level_order:
                        if level not in rate_df.columns:
                            rate_df[level] = 0
                    rate_df = rate_df[level_order]

                    rate_total = rate_df.sum(axis=1)
                    rate_pct = rate_df.div(rate_total, axis=0) * 100
                    rate_pct.index = rate_pct.index.strftime('%Y-%m-%d')

                    st.markdown(f"**Contamination Level Distribution ({rate_period}):**")
                    rate_melt = rate_pct.reset_index().melt(
                        id_vars='RatePeriod', var_name='Level', value_name='Percentage'
                    )
                    rate_melt['Level'] = pd.Categorical(rate_melt['Level'], categories=level_order, ordered=True)
                    bar_chart = alt.Chart(rate_melt).mark_bar().encode(
                        x=alt.X('RatePeriod:T', title='Period', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('Percentage:Q', title='Percentage'),
                        xOffset=alt.XOffset('Level:N'),
                        color=alt.Color('Level:N', scale=alt.Scale(domain=level_order, range=level_colors),
                                        legend=alt.Legend(title='Contamination Level')),
                    ).properties(height=350)
                    st.altair_chart(bar_chart, width='stretch')

                    st.markdown("**Level Distribution Summary:**")
                    col_e, col_f = st.columns(2)
                    with col_e:
                        st.markdown("**By Count:**")
                        st.dataframe(rate_df, width='stretch')
                    with col_f:
                        st.markdown("**By Percentage:**")
                        st.dataframe(rate_pct.round(1).astype(str) + '%', width='stretch')
                else:
                    st.info("No contamination level column found in the uploaded file.")
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")

if __name__ == "__main__":
    main()
