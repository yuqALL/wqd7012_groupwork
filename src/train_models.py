#!/usr/bin/env python3
"""
Train and save models for Streamlit Cloud deployment.
Run this script in an environment with Python 3.9-3.11 and numpy<2.0.

Models included:
- Random Forest
- XGBoost
- Hybrid NN-XGBoost

Example:
    pip install numpy scikit-learn xgboost pandas joblib torch
    python train_models.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import urllib.request
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

base_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "model")
DATA_URL = "https://raw.githubusercontent.com/yuqALL/wqd7012_groupwork/main/data/River_Water_Quality.csv"

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
   
def prepare_ml_data(df):
    df_base_clean = df.copy()

    # Format Date column
    df_base_clean['Date'] = pd.to_datetime(df_base_clean['Date'])

    # Drop rows values missing in target
    df_base_clean = df_base_clean.dropna(subset=['CCME_Values'])

    # Fill missing values for numeric columns with mean
    numeric_cols = df_base_clean.select_dtypes(include=['number']).columns
    df_base_clean[numeric_cols] = df_base_clean[numeric_cols].fillna(df_base_clean[numeric_cols].mean())

    # Drop the classification target to prevent target leakage
    if 'CCME_WQI' in df_base_clean.columns:
        df_base_clean = df_base_clean.drop(columns=['CCME_WQI'])

    df_baseline = df_base_clean.copy()
    # Drop unused columns
    cols_to_drop = ['Country', 'Waterbody Type', 'Date', 'Area']
    df_baseline = df_base_clean.drop(columns=[col for col in cols_to_drop if col in df_base_clean.columns])
    df_baseline.to_csv(os.path.join(data_dir, "River_Water_Quality_Final.csv"), index=False)

    return df_baseline

def train_rf_model(df_rf):

    if os.path.exists(os.path.join(models_dir, "rf_model.pkl")):
        print("✅ Trained Random Forest model found. Skipping training.")
        return None

    exclude_cols = ['CCME_Values']
    feature_cols = [col for col in df_rf.columns if col not in exclude_cols]

    X_rf = df_rf[feature_cols]
    y_rf = df_rf['CCME_Values']
    print(f"Random Forest Input - X shape: {X_rf.shape} | y shape: {y_rf.shape}")

    print("Building Random Forest pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_rf.columns)
        ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, max_depth=6, max_features=0.8,
                                            max_samples=0.8, random_state=42, n_jobs=-1))
    ])

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    print(f"Training Random Forest on {X_train_rf.shape[0]} samples...")

    # Fit model
    rf_pipeline.fit(X_train_rf, y_train_rf)
    print("Saving Random Forest model...")

    with open(os.path.join(models_dir, "rf_model.pkl"), 'wb') as f:
        pickle.dump(rf_pipeline, f)
    print(f"✅ Random Forest model saved to {os.path.join(models_dir, 'rf_model.pkl')}")

    # Predict on test set
    y_pred_rf = rf_pipeline.predict(X_test_rf)

    rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
    r2_rf = r2_score(y_test_rf, y_pred_rf)

    print(f"Random Forest Evaluation:")
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"MAE: {mae_rf:.4f}")
    print(f"R2 Score: {r2_rf:.4f}")

    return None

def train_xgb_model(df_xgb):

    if os.path.exists(os.path.join(models_dir, "xgb_model.pkl")):
        print("✅ Trained XGBoost model found. Skipping training.")
        return None

    exclude_cols = ['CCME_Values']
    feature_cols = [col for col in df_xgb.columns if col not in exclude_cols]
    X_xgb = df_xgb[feature_cols]
    y_xgb = df_xgb['CCME_Values']
    print(f"XGBoost Input - X shape: {X_xgb.shape} | y shape: {y_xgb.shape}")

    print("Building XGBoost pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_xgb.columns)
        ])

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   objective='reg:squarederror', n_jobs=-1))
    ])

    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
    print(f"Training XGBoost on {X_train_xgb.shape[0]} samples...")

    # Fit model
    xgb_pipeline.fit(X_train_xgb, y_train_xgb)

    print("Saving XGBoost model...")
    with open(os.path.join(models_dir, "xgb_model.pkl"), 'wb') as f:
        pickle.dump(xgb_pipeline, f)
    print(f"✅ XGBoost model saved to {os.path.join(models_dir, 'xgb_model.pkl')}")

    # Predict on test set
    y_pred_xgb = xgb_pipeline.predict(X_test_xgb)

    rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test_xgb, y_pred_xgb)
    r2_xgb = r2_score(y_test_xgb, y_pred_xgb)

    print(f"XGBoost Evaluation:")
    print(f"RMSE: {rmse_xgb:.4f}")
    print(f"MAE: {mae_xgb:.4f}")
    print(f"R2 Score: {r2_xgb:.4f}")

    return None

def prepare_hybrid_nn_xgb_data(df_hybrid):
    
    exclude_cols = ['CCME_Values']
    feature_cols = [col for col in df_hybrid.columns if col not in exclude_cols]
    
    X_hybrid = df_hybrid[feature_cols]
    y_hybrid = df_hybrid['CCME_Values']
    
    X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(X_hybrid, y_hybrid, test_size=0.2, random_state=42)
    print(f"Hybrid NN-XGBoost Input - X shape: {X_hybrid.shape} | y shape: {y_hybrid.shape}")

    scaler = StandardScaler()
    X_train_hybrid_scaled = scaler.fit_transform(X_train_hybrid)
    X_test_hybrid_scaled = scaler.transform(X_test_hybrid)
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_hybrid_scaled, y_train_hybrid, test_size=0.1, random_state=42)
    
    # Data Preparation
    # PyTorch NN requires (Samples, Features) input format

    # --- Training tensors ---
    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.float32).view(-1, 1)

    # --- Validation tensors ---
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # --- Test tensors ---
    X_test_tensor = torch.tensor(X_test_hybrid_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_hybrid.values, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader for batch training to optimize memory
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    train_loader_extract = DataLoader(train_dataset, batch_size=256, shuffle=False)  
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, train_loader_extract, val_loader, test_loader, y_train_final, y_test_hybrid

def train_hybrid_nn_xgb_model(df):
    print("Preparing data for Hybrid NN-XGBoost...")
    train_loader, train_loader_extract, val_loader, test_loader, y_train_final, y_test_hybrid = prepare_hybrid_nn_xgb_data(df)
    
    print("Initializing Hybrid model...")
    sample_batch, _ = next(iter(train_loader))
    num_features = sample_batch.shape[1]
    hybrid_nn_model = WaterQualityNN(num_features)

    if os.path.exists(os.path.join(models_dir, "best_hybrid_nn_model.pth")):
        print("✅ Trained NN model found.")
        hybrid_nn_model.load_state_dict(torch.load(os.path.join(models_dir, "best_hybrid_nn_model.pth")))

    else:
        # --- NN Training ---
        criterion = nn.MSELoss()
        optimizer = optim.Adam(hybrid_nn_model.parameters(), lr=0.001)
        epochs = 50
        patience = 5
        best_val_loss = float('inf')
        counter = 0
        
        print("Training NN model...")
        for epoch in range(epochs):
            # --- TRAINING ---
            hybrid_nn_model.train()
            train_loss = 0.0

            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
            
            for batch_X, batch_y in train_progress:
                optimizer.zero_grad()

                outputs, _ = hybrid_nn_model(batch_X)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_progress.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # --- VALIDATION ---
            hybrid_nn_model.eval()
            val_loss = 0.0

            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
            
            with torch.no_grad():
                for val_X, val_y in val_progress:

                    val_outputs, _ = hybrid_nn_model(val_X)

                    loss = criterion(val_outputs, val_y)
                    val_loss += loss.item()

                    val_progress.set_postfix(val_loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)

            # Epoch summary
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                counter = 0

                torch.save(hybrid_nn_model.state_dict(), os.path.join(models_dir, "best_hybrid_nn_model.pth"))
                print("Validation loss improved. Model saved.")

            else:
                counter += 1
                print(f"Early Stopping Counter: {counter}/{patience}")

                if counter >= patience:
                    print("Early stopping triggered.")
                    break
    
        print(f"✅ Hybrid NN model saved to {os.path.join(models_dir, 'best_hybrid_nn_model.pth')}")

    print("Preparing data for Hybrid NN-XGBoost...")

    # Load hybrid model
    hybrid_nn_model.load_state_dict(torch.load(os.path.join(models_dir, "best_hybrid_nn_model.pth")))
    
    # Extract Features & Train XGBoost
    print("Extracting NN features...")
    hybrid_nn_model.eval()

    # -- Train Features --
    train_features_list = []
    
    with torch.no_grad():
        for batch_X, _ in tqdm(train_loader_extract, desc="Extracting Train NN Features"):
            _, batch_features = hybrid_nn_model(batch_X)

            train_features_list.append(batch_features.cpu().numpy())

    train_hybrid_nn_xgb_features_np = np.vstack(train_features_list)

    # -- Test Features --
    test_features_list = []

    with torch.no_grad():
        for batch_X, _ in tqdm(test_loader, desc="Extracting Test NN Features"):
            _, batch_features = hybrid_nn_model(batch_X)

            test_features_list.append(batch_features.cpu().numpy())

    test_hybrid_nn_xgb_features_np = np.vstack(test_features_list)

    print(f"Extracted features shape: {test_hybrid_nn_xgb_features_np.shape}")

    if os.path.exists(os.path.join(models_dir, "best_hybrid_xgb_model.pkl")):
        print("✅ Trained Hybrid XGBoost model found. Skipping training.")
        return None
   
    # Train Hybrid NN-XGBoost Regressor
    print("Training Hybrid NN-XGBoost: ")

    hybrid_nn_xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, 
                              subsample=0.8, colsample_bytree=0.8, random_state=42, 
                              objective='reg:squarederror', n_jobs=-1)
    
    hybrid_nn_xgb_model.fit(train_hybrid_nn_xgb_features_np, y_train_final)

    with open(os.path.join(models_dir, "best_hybrid_nn_xgb_model.pkl"), 'wb') as f:
        pickle.dump(hybrid_nn_xgb_model, f)

    print(f"✅ Hybrid NN-XGBoost model saved to {os.path.join(models_dir, 'best_hybrid_nn_xgb_model.pkl')}")
    
    # Predict on test set
    y_pred_hybrid = hybrid_nn_xgb_model.predict(test_hybrid_nn_xgb_features_np)
    
    rmse_hybrid = np.sqrt(mean_squared_error(y_test_hybrid, y_pred_hybrid))
    mae_hybrid = mean_absolute_error(y_test_hybrid, y_pred_hybrid)
    r2_hybrid = r2_score(y_test_hybrid, y_pred_hybrid)

    print(f"Hybrid NN-XGBoost Evaluation:")
    print(f"RMSE: {rmse_hybrid:.4f}")
    print(f"MAE: {mae_hybrid:.4f}")
    print(f"R2 Score: {r2_hybrid:.4f}")
    
    return None

if __name__ == "__main__":

    RAW_DATA_PATH = os.path.join(data_dir, "River_Water_Quality.csv")
    FINAL_DATA_PATH = os.path.join(data_dir, "River_Water_Quality_Final.csv")

    if os.path.exists(FINAL_DATA_PATH):
        print("✅ Processed dataset already exists. Skipping data preparation.")
        River_Water_Quality_Final_df = pd.read_csv(FINAL_DATA_PATH)

    else:
        if not os.path.exists(RAW_DATA_PATH):
            print(f"Downloading dataset from {DATA_URL}...")
            urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
            print("✅ Raw Dataset downloaded")

        print("Loading dataset...")
        River_Water_Quality_df = pd.read_csv(RAW_DATA_PATH)
        print(f"Dataset loaded: {len(River_Water_Quality_df)} records")

        River_Water_Quality_Final_df = prepare_ml_data(River_Water_Quality_df)

    print("✅ Final processed dataset created")

    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    train_rf_model(River_Water_Quality_Final_df)

    print("\n" + "="*50)
    print("Training XGBoost model...")
    print("="*50)
    train_xgb_model(River_Water_Quality_Final_df)

    print("\n" + "="*50)
    print("Training Hybrid NN-XGBoost model...")
    print("="*50)
    train_hybrid_nn_xgb_model(River_Water_Quality_Final_df)

    print("\n" + "="*50)
    print("🎉 All models trained successfully!")
    print("="*50)
    print(f"Model files created:")
    print(f"  - {os.path.join(models_dir, 'rf_model.pkl')}")
    print(f"  - {os.path.join(models_dir, 'xgb_model.json')}")
    print(f"  - {os.path.join(models_dir, 'best_hybrid_nn_model.pth')}")
    print(f"  - {os.path.join(models_dir, 'best_hybrid_xgb_model.pkl')}")
    print("\nUpload these files to your GitHub repository.")