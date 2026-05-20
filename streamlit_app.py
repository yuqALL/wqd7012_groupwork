import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pickle
import urllib.request

from datetime import datetime, timedelta

# PyTorch imports for Deep Learning model
import torch
import torch.nn as nn

plt.rcParams['font.family'] = 'DejaVu Sans'

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "model")

RF_MODEL_PATH = os.path.join(model_dir, "rf_model.pkl")
XGB_MODEL_PATH = os.path.join(model_dir, "xgb_model.pkl")
NN_MODEL_PATH = os.path.join(model_dir, "best_hybrid_nn_model.pth") 
HYBRID_NN_XGB_MODEL_PATH = os.path.join(model_dir, "best_hybrid_nn_xgb.pkl")
MODEL_URL_RF = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/model/rf_model.pkl"
MODEL_URL_XGB = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/model/xgb_model.pkl"
MODEL_URL_NN = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/model/best_hybrid_nn_model.pth"
MODEL_URL_HYBRID_NN_XGB = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/model/best_hybrid_nn_xgb.pkl"

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
    if not os.path.exists(NN_MODEL_PATH):
        try:
            with st.spinner("Downloading Hybrid NN model..."):
                urllib.request.urlretrieve(MODEL_URL_NN, NN_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to download Hybrid NN model: {e}")
    if not os.path.exists(HYBRID_NN_XGB_MODEL_PATH):
        try:
            with st.spinner("Downloading Hybrid NN-XGBoost model..."):
                urllib.request.urlretrieve(MODEL_URL_HYBRID_NN_XGB, HYBRID_NN_XGB_MODEL_PATH)
        except Exception as e:
            st.warning(f"Failed to download Hybrid NN-XGBoost model: {e}")
            success = False
    return success

@st.cache_resource
def load_rf_model():
    with open(RF_MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_resource
def load_xgb_model():
    with open(XGB_MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    pipeline.named_steps['regressor'].set_params(n_jobs=1, predictor='cpu_predictor')
    return pipeline

def generate_random_timestamps(n):
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()

    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = np.random.randint(0, total_seconds, n)
    timestamps = [start_date + timedelta(seconds=int(sec)) for sec in random_seconds]

    return pd.to_datetime(timestamps)

# Neural Network Architecture for Hybrid NN-XGBoost model
class WaterQualityNN(nn.Module):
    def __init__(self, num_features):
        super(WaterQualityNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.regressor(features)

        return output, features

@st.cache_resource
def load_hybrid_models():

    # Load saved NN model
    state_dict = torch.load(NN_MODEL_PATH, map_location=torch.device('cpu'))
    num_features = state_dict['feature_extractor.0.weight'].shape[1]

    nn_model = WaterQualityNN(num_features)
    nn_model.load_state_dict(state_dict)
    nn_model.eval()

    # Load Hybrid NN-XGBoost model
    with open(HYBRID_NN_XGB_MODEL_PATH, 'rb') as f:
        hybrid_nn_xgb_model = pickle.load(f)

    return nn_model, hybrid_nn_xgb_model

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
            ["🌲 Random Forest", "🚀 XGBoost", "🧠 Hybrid NN-XGBoost"],
            help="Choose between Random Forest, XGBoost, or Hybrid NN-XGBoost model"
        )

    # ============================================
    # APPLICATION DEVELOPMENT
    # ============================================

    st.header("River Water Quality Prediction")

    st.markdown("""
    This application allows you to predict the Water Quality Index (WQI) based on uploaded water quality data.
    """)

    download_models()

    pipeline = None
    nn_model = None
    use_hybrid_nn_xgb = False

    if model_choice == "🌲 Random Forest":
        if os.path.exists(RF_MODEL_PATH):
            try:
                pipeline = load_rf_model()
                st.sidebar.success("✅ Loaded Random Forest model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load RF model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ RF model file not found on GitHub...")

    elif model_choice == "🚀 XGBoost":
        if os.path.exists(XGB_MODEL_PATH):
            try:
                pipeline = load_xgb_model()
                st.sidebar.success("✅ Loaded XGBoost model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load XGBoost model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ XGB model file not found on GitHub...")
            st.stop()
    else:
        use_hybrid_nn_xgb = True
        if os.path.exists(HYBRID_NN_XGB_MODEL_PATH):
            try:
                nn_model, hybrid_nn_xgb_model = load_hybrid_models()
                st.sidebar.success("✅ Loaded Hybrid NN-XGBoost model")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load Hybrid NN-XGBoost model: {str(e)}")
                st.stop()
        else:
            st.sidebar.error("❌ Hybrid NN-XGBoost model file not found on GitHub...")
            st.stop()

    st.markdown("### Input Water Quality Parameters")

    st.markdown("""
    **Expected Input Format:**
    Upload a CSV or Excel file with the following columns:
    """)

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

        st.markdown("**Example Input Format (First 3 rows from sample_input.csv):**")

        st.dataframe(sample_preview.style.format({
            'Dissolved Oxygen (mg/L)': '{:.2f}',
            'Nitrate (mg/L)': '{:.2f}',
            'Nitrogen (mg/L)': '{:.2f}',
            'Ammonia (mg/L)': '{:.2f}',
            'BOD (mg/L)': '{:.2f}',
            'Orthophosphate (mg/L)': '{:.2f}',
            'pH': '{:.2f}',
            'Temperature': '{:.2f}'
        }), width='stretch')

        with open(sample_csv_path, 'rb') as f:
            st.download_button(
                label="Download Sample Input CSV",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv"
            )
    else:
        st.markdown("**Example CSV content:**")
        sample_df = pd.DataFrame({
            'Dissolved Oxygen (mg/L)': [8.5, 7.2],
            'Nitrate (mg/L)': [5.2, 6.8],
            'Nitrogen (mg/L)': [2.1, 1.8],
            'Ammonia (mg/L)': [0.3, 0.5],
            'BOD (mg/L)': [4.5, 6.2],
            'Orthophosphate (mg/L)': [0.1, 0.15],
            'pH': [7.5, 7.8],
            'Temperature': [15.0, 18.5]
        })
        st.dataframe(sample_df, width='stretch')
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
            type=["csv", "xlsx"],
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
            pred_df = pd.read_csv(sample_csv_path)
            file_name = "sample_input.csv"
            
            st.session_state.pred_input_df = pred_df
            st.session_state.pred_input_name = file_name
            st.session_state.sample_loaded = True

            st.session_state.prediction_result_df = None
            st.session_state.prediction_display_df = None
            st.session_state.last_prediction_done = False

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

                    nn_means = None
                    nn_stds = None
                    if use_hybrid_nn_xgb and nn_model is not None:
                        nn_means = torch.tensor(
                            [0.45897955, 3.19264486, 10.03764867, 0.3192486,
                             7.76286298, 11.03018225, 4.72756508, 4.56060162],
                            dtype=torch.float32
                        )
                        nn_stds = torch.tensor(
                            [3.99296878, 10.60728528, 2.07854648, 1.2832782,
                             0.48097796, 3.99207888, 4.6257997, 4.78472973],
                            dtype=torch.float32
                        )
                        nn_model.eval()

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

                        if use_hybrid_nn_xgb and nn_model is not None:
                            nn_features = chunk[[
                                'Ammonia (mg/L)', 'BOD (mg/L)',
                                'Dissolved Oxygen (mg/L)', 'Orthophosphate (mg/L)',
                                'pH', 'Temperature',
                                'Nitrogen (mg/L)', 'Nitrate (mg/L)'
                            ]].values.astype(np.float32)

                            nn_features = (nn_features - nn_means.numpy()) / nn_stds.numpy()
                            input_tensor = torch.tensor(nn_features, dtype=torch.float32)

                            with torch.no_grad():
                                _, extracted_features = nn_model(input_tensor)

                            features_np = extracted_features.numpy()
                            chunk_preds = hybrid_nn_xgb_model.predict(features_np).tolist()
  
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
                        partial_df['Timestamp'] = generate_random_timestamps(len(partial_df))
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
                    result_df['Timestamp'] = generate_random_timestamps(len(result_df))
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
            "🧠 Hybrid NN-XGBoost": "hybrid"
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
            <h2 style="color: white; margin: 0;">Average WQI: {avg_wqi:.2f}</h2>
            <h3 style="color: white; margin: 10px 0 0 0;">Dominant Level: {mode_category}</h3>
            <p style="color: white; margin: 15px 0 0 0; font-size: 14px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Water Contamination Levels Categories")

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
    st.subheader("Water Quality Monitoring and Contamination Assessment Dashboard")

    st.markdown("""
    Upload prediction data to visualize water quality trends and contamination levels.
    The expected format is the output CSV from the prediction above.
    """)

    demo_csv_path = os.path.join(data_dir, "demo_dashboard.csv")
    st.markdown("**Upload Data*")

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

                    if period in ["Weekly", "Monthly"]:
                        selected_year = st.selectbox("Select Year", sorted(dash_df['Timestamp'].dt.year.unique()), key="trend_year")
                        trend_df = dash_df[dash_df['Timestamp'].dt.year == selected_year].copy()
                        
                    else:
                        trend_df = dash_df.copy()

                    if period == "Weekly":
                        month_options = {"January": 1, "February": 2, "March": 3, "April": 4,
                                         "May": 5, "June": 6, "July": 7, "August": 8,
                                         "September": 9, "October": 10, "November": 11, "December": 12} 
                        
                        selected_month_name = st.selectbox("Select Month", list(month_options.keys()), key="trend_month")
                        selected_month = month_options[selected_month_name]

                        trend_df = trend_df[trend_df['Timestamp'].dt.month == selected_month]
                        trend_df['Period'] = trend_df['Timestamp'].dt.to_period('W').dt.start_time

                    elif period == "Monthly":
                        trend_df['Period'] = trend_df['Timestamp'].dt.to_period('M').dt.start_time

                    else:
                        trend_df['Period'] = trend_df['Timestamp'].dt.to_period('Y').dt.start_time

                    avg_df = trend_df.groupby('Period')[selected_param].mean().reset_index()

                    if period == "Weekly":
                        weekly_labels = []

                        for start_date in avg_df['Period']:
                            end_date = start_date + pd.Timedelta(days=6)
                            label = (start_date.strftime('%d %b') + ' - ' + end_date.strftime('%d %b'))
                            weekly_labels.append(label)

                        avg_df['Label'] = weekly_labels

                    elif period == "Monthly":
                        avg_df['Label'] = avg_df['Period'].dt.strftime('%b')
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr',
                                       'May', 'Jun', 'Jul', 'Aug',
                                       'Sep', 'Oct', 'Nov', 'Dec']
                        
                        avg_df['Label'] = pd.Categorical(avg_df['Label'], categories=month_order, ordered=True)
                        avg_df = avg_df.sort_values('Label')

                    else:
                        avg_df['Label'] = avg_df['Period'].dt.strftime('%Y')

                    st.markdown(f"**Average {selected_param} ({period}):**")

                    line_chart = alt.Chart(avg_df).mark_line(point=True).encode(
                        x=alt.X('Label:N', title=period, sort=month_order if period == "Monthly" else None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y(f'{selected_param}:Q', title=selected_param),
                        tooltip=['Label', selected_param]
                    ).properties(height=400)

                    st.altair_chart(line_chart, width='stretch')
                   
                else:
                    st.info("No water quality parameter columns found in the uploaded file.")

                st.markdown("---")
                st.markdown("#### Contamination Level Rates")

                model_contam_map = {
                    "🌲 Random Forest": 'Contamination_RF',
                    "🚀 XGBoost": 'Contamination_XGB',
                    "🧠 Hybrid NN-XGBoost": 'Contamination_Hybrid',
                }
                model_wqi_map = {
                    "🌲 Random Forest": 'CCME_WQI_RF',
                    "🚀 XGBoost": 'CCME_WQI_XGB',
                    "🧠 Hybrid NN-XGBoost": 'CCME_WQI_Hybrid',
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
                    rate_period = st.selectbox("Select Period for Contamination Rates", ["Weekly", "Monthly", "Yearly"], key="rate_period")

                    level_order = ["Excellent", "Good", "Fair", "Marginal", "Poor"]
                    level_colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]

                    # Weekly and Monthly require year filtering
                    if rate_period in ["Weekly", "Monthly"]:

                        selected_year = st.selectbox("Select Year", sorted(dash_df['Timestamp'].dt.year.unique()), key="selected_year")
                        filtered_df = dash_df[dash_df['Timestamp'].dt.year == selected_year].copy()

                    else:
                        filtered_df = dash_df.copy()

                    if rate_period == "Weekly":
                        month_options = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                                         "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

                        selected_month_name = st.selectbox("Select Month", list(month_options.keys()), key="selected_month")
                        selected_month = month_options[selected_month_name]

                        filtered_df = filtered_df[filtered_df['Timestamp'].dt.month == selected_month]
                        filtered_df['RatePeriod'] = filtered_df['Timestamp'].dt.to_period('W').dt.start_time

                        x_title = "Week"
            
                    elif rate_period == "Monthly":
                        filtered_df['RatePeriod'] = filtered_df['Timestamp'].dt.to_period('M').dt.start_time
                        x_title = "Month"

                    else:
                        filtered_df['RatePeriod'] = filtered_df['Timestamp'].dt.to_period('Y').dt.start_time
                        x_title = "Year"

                    rate_df = filtered_df.groupby('RatePeriod')[contam_col].value_counts().unstack(fill_value=0)

                    for level in level_order:
                        if level not in rate_df.columns:
                            rate_df[level] = 0

                    rate_df = rate_df[level_order]
                    rate_total = rate_df.sum(axis=1)
                    rate_pct = rate_df.div(rate_total, axis=0) * 100

                    if rate_period == "Weekly":
                        weekly_ranges = []

                        for start_date in pd.to_datetime(rate_pct.index):
                            end_date = start_date + pd.Timedelta(days=6)

                            label = (start_date.strftime('%d %b') + ' - ' + end_date.strftime('%d %b'))
                            weekly_ranges.append(label)

                        rate_pct.index = weekly_ranges
                        rate_pct.index.name = "RatePeriod"

                    elif rate_period == "Monthly":
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr',
                                       'May', 'Jun', 'Jul', 'Aug',
                                       'Sep', 'Oct', 'Nov', 'Dec']
                        rate_pct.index = pd.to_datetime(rate_pct.index).strftime('%b')

                        rate_pct = rate_pct.reindex(month_order)
                        rate_pct = rate_pct.dropna(how='all')

                    else:
                        rate_pct.index = pd.to_datetime(rate_pct.index).strftime('%Y')

                    st.markdown(f"**Contamination Level Distribution ({rate_period}):**")

                    rate_pct_reset = rate_pct.reset_index()
                    rate_pct_reset['SortOrder'] = range(len(rate_pct_reset))
                    rate_melt = rate_pct_reset.melt(id_vars=['RatePeriod', 'SortOrder'], var_name='Level', value_name='Percentage')
                    rate_melt['Level'] = pd.Categorical(rate_melt['Level'], categories=level_order, ordered=True)

                    if rate_period == "Weekly":
                        chart = alt.Chart(rate_melt).mark_bar().encode(
                        x=alt.X('RatePeriod:N', title=x_title, sort=alt.SortField(field='SortOrder'), axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Percentage:Q', stack='normalize', title='Percentage'),
                        color=alt.Color('Level:N', scale=alt.Scale(domain=level_order, range=level_colors),
                                        legend=alt.Legend(title='Contamination Level')),
                    ).properties(height=400)
                        
                    else:

                        if rate_period == "Monthly":
                            sort_order = ['Jan', 'Feb', 'Mar', 'Apr',
                                          'May', 'Jun', 'Jul', 'Aug',
                                          'Sep', 'Oct', 'Nov', 'Dec']
                            
                        elif rate_period == "Yearly":
                            sort_order = sorted(rate_melt['RatePeriod'].unique())
                        
                        else:
                            sort_order = None

                        chart = alt.Chart(rate_melt).mark_bar().encode(
                            x=alt.X('RatePeriod:N', title=x_title, sort=sort_order, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y('Percentage:Q', title='Percentage'),
                            xOffset = alt.XOffset('Level:N'),
                            color=alt.Color('Level:N', scale=alt.Scale(domain=level_order, range=level_colors),
                                            legend=alt.Legend(title='Contamination Level')),
                        ).properties(height=350)

                    st.altair_chart(chart, width='stretch')

                    st.markdown("**Level Distribution Summary:**")
                    col_e, col_f = st.columns(2)

                    summary_count_df = rate_df.copy()
                    summary_pct_df = rate_pct.copy()

                    if rate_period == "Weekly":
                        weekly_ranges = []

                        for start_date in pd.to_datetime(summary_count_df.index):
                            end_date = start_date + pd.Timedelta(days=6)
                            label = (start_date.strftime('%d %b') + ' - ' + end_date.strftime('%d %b'))
                            weekly_ranges.append(label)
                        
                        summary_count_df.index = weekly_ranges
                        summary_pct_df.index = weekly_ranges

                    elif rate_period == "Monthly":
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr',
                                       'May', 'Jun', 'Jul', 'Aug',
                                       'Sep', 'Oct', 'Nov', 'Dec']
                        
                        summary_count_df.index = pd.Index(pd.to_datetime(summary_count_df.index).strftime('%b'))
                        summary_pct_df.index = summary_pct_df.index.astype(str)

                        summary_count_df = summary_count_df.reindex(month_order).dropna(how='all')
                        summary_pct_df = summary_pct_df.reindex(month_order).dropna(how='all')

                    else:
                        summary_count_df.index = pd.to_datetime(summary_count_df.index).strftime('%Y')
                        summary_pct_df.index = pd.to_datetime(summary_pct_df.index).strftime('%Y')

                    # Rename index column
                    summary_count_df.index.name = "Period"
                    summary_pct_df.index.name = "Period"

                    summary_pct_df = summary_pct_df.round(1).astype(str) + '%'

                    with col_e:
                        st.markdown("**By Count:**")
                        st.dataframe(summary_count_df, width='stretch')

                    with col_f:
                        st.markdown("**By Percentage:**")
                        st.dataframe(summary_pct_df, width='stretch')
                else:
                    st.info("No contamination level column found in the uploaded file.")

        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")

if __name__ == "__main__":
    main()
