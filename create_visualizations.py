#!/usr/bin/env python3
"""
Create visualizations for the ML comparison results
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib with error handling
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib imported successfully!")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"❌ Matplotlib import failed: {e}")
    print("Will create text-based visualizations instead.")

def create_ml_features(data, target_col='Appliances', lag_periods=[1, 2, 3, 6, 12, 24]):
    """Create features for traditional ML models from time series data"""
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
        df_ml['hour'] = 12
        df_ml['day_of_week'] = 1
        df_ml['day_of_month'] = 1
        df_ml['month'] = 1
    
    # Create cyclical features
    df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['hour'] / 24)
    df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['hour'] / 24)
    df_ml['day_sin'] = np.sin(2 * np.pi * df_ml['day_of_week'] / 7)
    df_ml['day_cos'] = np.cos(2 * np.pi * df_ml['day_of_week'] / 7)
    
    return df_ml

def create_text_visualizations(results_df, predictions, y_test):
    """Create text-based visualizations when matplotlib is not available"""
    print("\n" + "="*60)
    print("📊 TEXT-BASED VISUALIZATIONS")
    print("="*60)
    
    # 1. Performance comparison bar chart (text)
    print("\n📈 MODEL PERFORMANCE COMPARISON (Text Bar Chart)")
    print("-" * 50)
    
    metrics = ['MAE', 'MAPE', 'RMSE', 'R2']
    for metric in metrics:
        print(f"\n{metric}:")
        max_val = results_df[metric].max()
        for model, value in results_df[metric].items():
            # Create a simple text bar
            bar_length = int((value / max_val) * 30)
            bar = "█" * bar_length
            print(f"  {model:<15}: {bar:<30} {value:.4f}")
    
    # 2. Predictions vs Actual (first 20 points)
    print("\n📊 PREDICTIONS VS ACTUAL (First 20 points)")
    print("-" * 50)
    print("Point  | Actual | RF Pred | LR Pred | SVR Pred")
    print("-" * 50)
    
    for i in range(min(20, len(y_test))):
        actual = y_test.iloc[i]
        rf_pred = predictions['Random Forest'][i]
        lr_pred = predictions['Linear Regression'][i]
        svr_pred = predictions['SVR'][i]
        print(f"{i:6d} | {actual:6.1f} | {rf_pred:7.1f} | {lr_pred:7.1f} | {svr_pred:7.1f}")
    
    # 3. Error analysis
    print("\n📉 ERROR ANALYSIS")
    print("-" * 30)
    for model_name, pred in predictions.items():
        errors = y_test - pred
        print(f"\n{model_name}:")
        print(f"  Mean Error: {errors.mean():.2f}")
        print(f"  Std Error:  {errors.std():.2f}")
        print(f"  Min Error:  {errors.min():.2f}")
        print(f"  Max Error:  {errors.max():.2f}")

def create_matplotlib_visualizations(results_df, predictions, y_test, rf_model, X_train):
    """Create matplotlib visualizations"""
    print("\n🎨 Creating matplotlib visualizations...")
    
    plt.figure(figsize=(16, 12))
    
    # 1. Metrics comparison bar chart
    plt.subplot(2, 3, 1)
    metrics = ['MAE', 'MAPE', 'RMSE']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (model_name, model_results) in enumerate(results_df.iterrows()):
        values = [model_results[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.yscale('log')
    
    # 2. R² comparison
    plt.subplot(2, 3, 2)
    r2_values = results_df['R2'].values
    model_names = results_df.index.tolist()
    colors = ['skyblue', 'lightgreen', 'salmon']
    bars = plt.bar(model_names, r2_values, alpha=0.8, color=colors)
    plt.ylabel('R² Score')
    plt.title('R² Comparison (Higher is Better)')
    plt.xticks(rotation=45)
    for i, (bar, value) in enumerate(zip(bars, r2_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Predictions vs Actual (first 200 points)
    plt.subplot(2, 3, 3)
    n_points = min(200, len(y_test))
    x_range = range(n_points)
    plt.plot(x_range, y_test.iloc[:n_points], label='Actual', alpha=0.7, linewidth=1)
    for model_name, pred in predictions.items():
        plt.plot(x_range, pred[:n_points], label=f'{model_name} Predicted', alpha=0.7, linewidth=1)
    plt.xlabel('Time Steps')
    plt.ylabel('Energy Consumption')
    plt.title('Predictions vs Actual (First 200 points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Feature importance (Random Forest)
    plt.subplot(2, 3, 4)
    feature_importance = rf_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names, 
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    
    # 5. Error distribution
    plt.subplot(2, 3, 5)
    for i, (model_name, pred) in enumerate(predictions.items()):
        errors = y_test - pred
        plt.hist(errors, bins=30, alpha=0.6, label=model_name, density=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Residual plot (Random Forest)
    plt.subplot(2, 3, 6)
    rf_residuals = y_test - predictions['Random Forest']
    plt.scatter(predictions['Random Forest'], rf_residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Random Forest)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_comparison_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'ml_comparison_visualizations.png'")
    plt.show()

def main():
    print("🎯 Creating Visualizations for ML Model Comparison")
    print("=" * 60)
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    df = pd.read_csv('energydata_complete.csv', index_col=0, parse_dates=True)
    df_ml = create_ml_features(df)
    df_ml_clean = df_ml.dropna()
    
    feature_cols = [col for col in df_ml_clean.columns if col != 'Appliances']
    X_ml = df_ml_clean[feature_cols]
    y_ml = df_ml_clean['Appliances']
    
    # Split data
    test_size = 30 * 144
    X_train = X_ml.iloc[:-test_size]
    X_test = X_ml.iloc[-test_size:]
    y_train = y_ml.iloc[:-test_size]
    y_test = y_ml.iloc[-test_size:]
    
    # Train models
    print("🤖 Training models...")
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    scaler_standard = StandardScaler()
    X_train_scaled = scaler_standard.fit_transform(X_train)
    X_test_scaled = scaler_standard.transform(X_test)
    
    results = {}
    predictions = {}
    
    for model_name, model in models.items():
        if model_name in ['Linear Regression', 'SVR']:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        mape = mean_absolute_percentage_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        
        results[model_name] = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        predictions[model_name] = pred
    
    results_df = pd.DataFrame(results).T
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        create_matplotlib_visualizations(results_df, predictions, y_test, models['Random Forest'], X_train)
    else:
        create_text_visualizations(results_df, predictions, y_test)
    
    print("\n✅ Visualization creation completed!")

if __name__ == "__main__":
    main()

