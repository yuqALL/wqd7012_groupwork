"""
Generate all visualization images for the Streamlit app.
Images are generated to match the notebook (OCC3_AML_Group13_GroupProject.ipynb) exactly.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "River_Water_Quality.csv"
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found. Please ensure the data file is in the current directory.")
    sys.exit(1)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df[(df['Date'].dt.year >= 2000) & (df['Date'].dt.year <= 2023)].copy()
df_sample = df.sample(n=10000, random_state=42)
print(f"Data loaded: {len(df)} records (2000-2023)")

# ============================================================
# 1. EDA Distribution Plot (matches notebook EDA 1)
# ============================================================
print("Generating EDA distribution plot...")
sns.set_style("whitegrid")

cols_to_plot = ['Ammonia (mg/l)', 'Dissolved Oxygen (mg/l)', 'Orthophosphate (mg/l)']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, col in enumerate(cols_to_plot):
    if col in df_sample.columns:
        sns.histplot(df_sample[col], kde=True, ax=axes[i], color='skyblue', log_scale=True)
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel('Concentration (log scale)')
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  -> eda_distribution.png saved")

# ============================================================
# 2. EDA Time Series Plot (matches notebook EDA 2)
# ============================================================
print("Generating EDA time series plot...")
df['Year'] = df['Date'].dt.year
yearly_country_avg = df.groupby(['Year', 'Country'])[['Ammonia (mg/l)', 'Dissolved Oxygen (mg/l)', 'Orthophosphate (mg/l)']].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)

sns.lineplot(data=yearly_country_avg, x='Year', y='Ammonia (mg/l)',
             hue='Country', marker='o', ax=axes[0])
axes[0].set_title('Average Ammonia Levels by Country', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Ammonia (mg/l)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.3)

sns.lineplot(data=yearly_country_avg, x='Year', y='Dissolved Oxygen (mg/l)',
             hue='Country', marker='D', ax=axes[1])
axes[1].set_title('Average Dissolved Oxygen by Country', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Dissolved Oxygen (mg/l)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.3)

sns.lineplot(data=yearly_country_avg, x='Year', y='Orthophosphate (mg/l)',
             hue='Country', marker='s', ax=axes[2])
axes[2].set_title('Average Orthophosphate Levels by Country', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Year', fontsize=12)
axes[2].set_ylabel('Orthophosphate (mg/l)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Temporal Trends of Water Quality Parameters by Country', fontsize=18, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_timeseries.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  -> eda_timeseries.png saved")

# ============================================================
# 3. EDA Correlation Heatmap (matches notebook EDA 3)
# ============================================================
print("Generating EDA correlation heatmap...")
selected_features = ['Ammonia (mg/l)', 'Biochemical Oxygen Demand (mg/l)', 'Dissolved Oxygen (mg/l)',
                     'Orthophosphate (mg/l)', 'Nitrate (mg/l)', 'Nitrogen (mg/l)',
                     'Temperature', 'pH', 'CCME_Values']
selected_features = [col for col in selected_features if col in df.columns]
subset_df = df[selected_features]
corr_matrix = subset_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Water Quality Parameters Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  -> eda_correlation.png saved")

# ============================================================
# 4. Model Comparison Plot (matches notebook: 3 models)
# ============================================================
print("Generating model comparison plot...")
comparison_data = {
    'Model': ['Random Forest', 'XGBoost', 'Hybrid CNN-XGBoost'],
    'RMSE': [0.1507, 0.5017, 5.3968],
    'MAE': [0.0127, 0.1757, 3.1332],
    'R2 Score': [0.9999, 0.9986, 0.5559]
}
df_compare = pd.DataFrame(comparison_data)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ['royalblue', 'cornflowerblue', 'crimson']

sns.barplot(x='Model', y='RMSE', data=df_compare, ax=axes[0], palette=colors, hue='Model', legend=False)
axes[0].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')

sns.barplot(x='Model', y='MAE', data=df_compare, ax=axes[1], palette=colors, hue='Model', legend=False)
axes[1].set_title('MAE Comparison (Lower is Better)', fontweight='bold')

sns.barplot(x='Model', y='R2 Score', data=df_compare, ax=axes[2], palette=colors, hue='Model', legend=False)
axes[2].set_title('R² Score Comparison (Higher is Better)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  -> model_comparison.png saved")

print("\nAll images generated successfully!")