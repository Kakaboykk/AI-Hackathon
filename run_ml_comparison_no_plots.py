#!/usr/bin/env python3
"""
Run the ML comparison without matplotlib to avoid import issues
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("Running Appliances Energy Prediction: Model Comparison (RF, Linear, SVR)")

# Load the dataset
df = pd.read_csv('energydata_complete.csv', index_col=0, parse_dates=True)

# Feature engineering function
def create_ml_features(data, target_col='Appliances', lag_periods=[1, 2, 3, 6, 12, 24]):
    """
    Create features for traditional ML models from time series data
    """
    df_ml = data.copy()
    
    # Create lagged features for the target variable
    for lag in lag_periods:
        df_ml[f'{target_col}_lag_{lag}'] = df_ml[target_col].shift(lag)
    
    # Create rolling statistics for the target variable
    for window in [6, 12, 24]:
        df_ml[f'{target_col}_mean_{window}'] = df_ml[target_col].rolling(window=window).mean()
        df_ml[f'{target_col}_std_{window}'] = df_ml[target_col].rolling(window=window).std()
        df_ml[f'{target_col}_max_{window}'] = df_ml[target_col].rolling(window=window).max()
        df_ml[f'{target_col}_min_{window}'] = df_ml[target_col].rolling(window=window).min()
    
    # Create time-based features
    if hasattr(df_ml.index, 'hour'):
        df_ml['hour'] = df_ml.index.hour
        df_ml['day_of_week'] = df_ml.index.dayofweek
        df_ml['day_of_month'] = df_ml.index.day
        df_ml['month'] = df_ml.index.month
    else:
        # If index is not datetime, create dummy time features
        df_ml['hour'] = 12  # Default hour
        df_ml['day_of_week'] = 1  # Default day
        df_ml['day_of_month'] = 1  # Default day
        df_ml['month'] = 1  # Default month
    
    # Create cyclical features
    df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['hour'] / 24)
    df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['hour'] / 24)
    df_ml['day_sin'] = np.sin(2 * np.pi * df_ml['day_of_week'] / 7)
    df_ml['day_cos'] = np.cos(2 * np.pi * df_ml['day_of_week'] / 7)
    
    return df_ml

# Create features for ML models
df_ml = create_ml_features(df)

# Remove rows with NaN values and prepare data
df_ml_clean = df_ml.dropna()
feature_cols = [col for col in df_ml_clean.columns if col != 'Appliances']
X_ml = df_ml_clean[feature_cols]
y_ml = df_ml_clean['Appliances']

# silently prepare features and target

# Temporal split: Use last 30 days for testing
test_size = 30 * 144  # 30 days * 144 samples per day (10-minute intervals)

X_train = X_ml.iloc[:-test_size]
X_test = X_ml.iloc[-test_size:]
y_train = y_ml.iloc[:-test_size]
y_test = y_ml.iloc[-test_size:]

# Scale features for Linear Regression and SVR
scaler_standard = StandardScaler()
X_train_scaled = scaler_standard.fit_transform(X_train)
X_test_scaled = scaler_standard.transform(X_test)

#

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
}

# Store results
results = {}
predictions = {}

# Train all models and calculate metrics
for model_name, model in models.items():
    # Choose appropriate data (scaled for Linear Regression and SVR)
    if model_name in ['Linear Regression', 'SVR']:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:  # Random Forest
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    
    results[model_name] = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    predictions[model_name] = pred

# Create results comparison table
print("\nMODEL COMPARISON RESULTS")
print("=" * 60)
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df)

print("\nBEST PERFORMING MODELS:")
print("=" * 30)
for metric in ['MAE', 'MAPE', 'RMSE', 'R2']:
    if metric == 'R2':
        best_model = results_df[metric].idxmax()
        best_score = results_df[metric].max()
    else:
        best_model = results_df[metric].idxmin()
        best_score = results_df[metric].min()
    print(f"{metric}: {best_model} ({best_score:.4f})")

