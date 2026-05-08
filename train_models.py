#!/usr/bin/env python3
"""
Train and save models for Streamlit Cloud deployment.
Run this script in an environment with Python 3.9-3.11 and numpy<2.0.

Models included:
- Random Forest
- XGBoost
- Hybrid CNN (PyTorch)
- Hybrid CNN-XGBoost

Example:
    pip install numpy scikit-learn xgboost pandas joblib torch
    python train_models.py
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

RF_MODEL_PATH = "rf_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
CNN_MODEL_PATH = "best_hybrid_cnn.pth"
HYBRID_XGB_MODEL_PATH = "hybrid_xgb_model.pkl"

def prepare_ml_data(df):
    df_cleaned = df.copy()
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned['Year'] = df_cleaned['Date'].dt.year
    df_cleaned['Month'] = df_cleaned['Date'].dt.month
    cols_to_drop = ['Date', 'CCME_WQI', 'Area']
    df_cleaned = df_cleaned.drop(columns=[col for col in cols_to_drop if col in df_cleaned.columns])
    df_cleaned = df_cleaned.dropna(subset=['CCME_Values'])
    return df_cleaned

def train_rf_model(df):
    print("Preparing data...")
    df_cleaned = prepare_ml_data(df)
    df_rf = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_rf.drop(columns=[target_col])
    y = df_rf[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    print("Building Random Forest pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)

    print("Saving Random Forest model...")
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_pipeline, f)
    print(f"✅ Random Forest model saved to {RF_MODEL_PATH}")

    return rf_pipeline

def train_xgb_model(df):
    print("Preparing data...")
    df_cleaned = prepare_ml_data(df)
    df_xgb = df_cleaned.sample(n=min(20000, len(df_cleaned)), random_state=42)

    target_col = 'CCME_Values'
    X = df_xgb.drop(columns=[target_col])
    y = df_xgb[target_col]

    cat_features = ['Country', 'Waterbody Type']
    num_features = [col for col in X.columns if col not in cat_features]

    print("Building XGBoost pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0))
    ])

    print("Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_pipeline.fit(X_train, y_train)

    print("Saving XGBoost model...")
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_pipeline, f)
    print(f"✅ XGBoost model saved to {XGB_MODEL_PATH}")

    return xgb_pipeline


class WaterQualityCNN(nn.Module):
    def __init__(self, num_features):
        super(WaterQualityCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.fc_features = nn.Linear(64, 32)

        self.fc_output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)

        features = self.dropout(self.relu(self.fc_features(x)))

        output = self.fc_output(features)

        return output, features


def prepare_cnn_data(df):
    df_cleaned = prepare_ml_data(df)
    
    exclude_cols = ['Country', 'Waterbody Type', 'Year', 'Month', 'CCME_Values']
    feature_cols = [col for col in df_cleaned.columns if col not in exclude_cols]
    
    X = df_cleaned[feature_cols]
    y = df_cleaned['CCME_Values']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42
    )
    
    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    
    y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    
    return train_loader, val_loader, test_loader, scaler, X_test_tensor, y_test


def train_hybrid_cnn_model(df):
    print("Preparing data for Hybrid CNN...")
    train_loader, val_loader, test_loader, scaler, X_test_tensor, y_test = prepare_cnn_data(df)
    
    print("Initializing Hybrid CNN model...")
    num_features = 8
    hybrid_model = WaterQualityCNN(num_features)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)
    epochs = 50
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    
    print("Training Hybrid CNN model...")
    for epoch in range(epochs):
        hybrid_model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = hybrid_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        hybrid_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs, _ = hybrid_model(val_X)
                loss = criterion(val_outputs, val_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(hybrid_model.state_dict(), CNN_MODEL_PATH)
            print("Validation loss improved. Model saved.")
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    print(f"✅ Hybrid CNN model saved to {CNN_MODEL_PATH}")
    
    hybrid_model.eval()
    with torch.no_grad():
        predictions, _ = hybrid_model(X_test_tensor)
        predictions = predictions.numpy().flatten()
        y_test_np = y_test.values
    
    rmse = np.sqrt(mean_squared_error(y_test_np, predictions))
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    
    print(f"Hybrid CNN Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return hybrid_model, scaler


def train_hybrid_xgb_model(df, cnn_model=None):
    if cnn_model is None:
        print("Loading Hybrid CNN model for feature extraction...")
        num_features = 8
        cnn_model = WaterQualityCNN(num_features)
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        cnn_model.eval()
    
    print("Preparing data for Hybrid CNN-XGBoost...")
    train_loader, val_loader, test_loader, scaler, X_test_tensor, y_test = prepare_cnn_data(df)
    
    print("Extracting CNN features...")
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            _, features = cnn_model(batch_X)
            train_features.append(features.numpy())
            train_labels.append(batch_y.numpy())
    
    train_features = np.vstack(train_features)
    train_labels = np.vstack(train_labels).flatten()
    
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            _, features = cnn_model(batch_X)
            test_features.append(features.numpy())
            test_labels.append(batch_y.numpy())
    
    test_features = np.vstack(test_features)
    test_labels = np.vstack(test_labels).flatten()
    
    print(f"Extracted features shape: {train_features.shape}")
    
    print("Training Hybrid XGBoost model...")
    hybrid_xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    hybrid_xgb.fit(train_features, train_labels)
    
    print(f"Saving Hybrid XGBoost model to {HYBRID_XGB_MODEL_PATH}...")
    with open(HYBRID_XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(hybrid_xgb, f)
    print(f"✅ Hybrid XGBoost model saved to {HYBRID_XGB_MODEL_PATH}")
    
    y_pred = hybrid_xgb.predict(test_features)
    
    rmse = np.sqrt(mean_squared_error(test_labels, y_pred))
    mae = mean_absolute_error(test_labels, y_pred)
    r2 = r2_score(test_labels, y_pred)
    
    print(f"Hybrid CNN-XGBoost Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return hybrid_xgb


if __name__ == "__main__":
    import urllib.request

    DATA_URL = "https://raw.githubusercontent.com/24236510-ui/wqd7012_groupwork/main/River_Water_Quality.csv"
    DATA_PATH = "River_Water_Quality.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("✅ Dataset downloaded")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} records")

    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    train_rf_model(df)

    print("\n" + "="*50)
    print("Training XGBoost model...")
    print("="*50)
    train_xgb_model(df)

    print("\n" + "="*50)
    print("Training Hybrid CNN model...")
    print("="*50)
    train_hybrid_cnn_model(df)

    print("\n" + "="*50)
    print("Training Hybrid CNN-XGBoost model...")
    print("="*50)
    train_hybrid_xgb_model(df)

    print("\n" + "="*50)
    print("🎉 All models trained successfully!")
    print("="*50)
    print(f"Model files created:")
    print(f"  - {RF_MODEL_PATH}")
    print(f"  - {XGB_MODEL_PATH}")
    print(f"  - {CNN_MODEL_PATH}")
    print(f"  - {HYBRID_XGB_MODEL_PATH}")
    print("\nUpload these files to your GitHub repository.")