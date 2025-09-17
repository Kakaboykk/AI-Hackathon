# 🏠 Appliances Energy Prediction Project

A comprehensive machine learning project for predicting household energy consumption using LSTM neural networks and traditional ML algorithms. This project analyzes energy data from smart home sensors to forecast appliance energy usage patterns.

## 📊 Project Overview

This project implements multiple approaches to energy consumption prediction:
- **LSTM Neural Networks** for time series forecasting
- **Traditional ML Algorithms** (Random Forest, Linear Regression, SVR) for comparison
- **Feature Engineering** with lagged variables and rolling statistics
- **Comprehensive Evaluation** with multiple metrics

## 🗂️ Project Structure

```
├── 📁 Data Files
│   ├── energydata_complete.csv          # Main dataset (19,737 samples)
│   └── forecast_data.csv               # Generated forecast results
│
├── 🧠 Model Files
│   ├── appliances_energy_predictor.keras    # Trained LSTM model
│   ├── best_energy_predictor_model.pkl      # Best ML model (Random Forest)
│   └── feature_scaler.pkl                   # Feature scaler for ML models
│
├── 📓 Jupyter Notebooks
│   ├── appliances-energy-prediction-ml-comparison.ipynb    # ML comparison analysis
│   └── appliances-energy-prediction-using-lstms.ipynb      # LSTM implementation
│
├── 🐍 Python Scripts
│   ├── run_notebook_minimal.py              # ⭐ MAIN SCRIPT - LSTM without plots
│   ├── run_ml_comparison_no_plots.py        # ML comparison without plots
│   ├── run_notebook.py                      # Full LSTM with visualizations
│   └── create_visualizations.py             # Visualization utilities
│
└── 📄 Documentation
    └── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib tqdm joblib
```

### Running the Project

#### 1. **LSTM Energy Prediction** (Recommended)
```bash
python run_notebook_minimal.py
```
This script:
- ✅ Trains LSTM model on energy data
- ✅ Generates 30-day forecasts
- ✅ Saves trained model and forecast data
- ✅ No plotting dependencies

#### 2. **ML Algorithm Comparison**
```bash
python run_ml_comparison_no_plots.py
```
This script:
- ✅ Compares Random Forest, Linear Regression, and SVR
- ✅ Shows feature importance analysis
- ✅ Saves the best performing model

#### 3. **Full Analysis with Visualizations**
```bash
python run_notebook.py
```
This script:
- ✅ Complete LSTM analysis with plots
- ✅ Generates training loss charts
- ✅ Creates forecast visualizations
- ⚠️ Requires matplotlib display support

## 📈 Dataset Information

### Energy Data (`energydata_complete.csv`)
- **Size**: 19,737 samples (4.5 months of data)
- **Frequency**: 10-minute intervals
- **Features**: 28 variables including:
  - `Appliances`: Target variable (energy consumption)
  - `lights`: Light energy consumption
  - `T1-T9`: Temperature sensors in different rooms
  - `RH_1-RH_9`: Relative humidity sensors
  - `T_out`: Outside temperature
  - `Windspeed`: Wind speed
  - `Visibility`: Weather visibility
  - And more environmental sensors

### Data Preprocessing
- **Temporal Split**: Last 30 days (4,320 samples) for testing
- **Feature Scaling**: MinMaxScaler for LSTM, StandardScaler for ML
- **Time Series**: 288-step sequences (2 days of 10-min intervals)

## 🧠 Model Architectures

### LSTM Model
```python
Sequential([
    LSTM(units=200, activation='relu', input_shape=(288, 28)),
    Dropout(0.3),
    Dense(units=28)  # Multi-output prediction
])
```

### ML Models
- **Random Forest**: 100 estimators, parallel processing
- **Linear Regression**: With feature scaling
- **Support Vector Regression**: RBF kernel, optimized parameters

## 📊 Performance Metrics

The project evaluates models using:
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)

## 🔧 Feature Engineering

### For ML Models
- **Lagged Features**: 1, 2, 3, 6, 12, 24-step lags
- **Rolling Statistics**: Mean, std, max, min over 6, 12, 24 periods
- **Time Features**: Hour, day of week, month
- **Cyclical Encoding**: Sine/cosine transformations for time features

### For LSTM
- **Time Series Sequences**: 288-step windows
- **Multi-variate Input**: All 28 features as input
- **Multi-step Forecasting**: 30-day ahead predictions

## 📈 Generated Outputs

### Files Created
- `appliances_energy_predictor.keras` - Trained LSTM model
- `best_energy_predictor_model.pkl` - Best ML model
- `feature_scaler.pkl` - Feature scaler
- `forecast_data.csv` - 30-day forecast results
- `training_losses.png` - Training progress (if plots enabled)
- `appliances_forecast.png` - Forecast visualization (if plots enabled)

### Console Output
- Dataset information and statistics
- Model training progress
- Performance metrics
- Feature importance analysis
- Forecast statistics

## 🎯 Use Cases

This project can be used for:
- **Smart Home Optimization**: Predict energy usage patterns
- **Cost Management**: Forecast electricity bills
- **Load Balancing**: Optimize appliance usage timing
- **Anomaly Detection**: Identify unusual consumption patterns
- **Energy Efficiency**: Guide home automation systems

## 🔬 Technical Details

### LSTM Implementation
- **Sequence Length**: 288 steps (2 days)
- **Batch Size**: 16
- **Epochs**: 10 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: MSE

### ML Implementation
- **Cross-validation**: Temporal split (no data leakage)
- **Feature Selection**: Automated feature importance
- **Hyperparameter Tuning**: Optimized for energy prediction
- **Ensemble Methods**: Random Forest for robust predictions

## 📋 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
tqdm>=4.62.0
joblib>=1.1.0
```

## 🚨 Important Notes

- **Large Files**: `best_energy_predictor_model.pkl` is 54MB (exceeds GitHub's 50MB recommendation)
- **Memory Usage**: LSTM training requires ~4GB RAM
- **Training Time**: ~10-15 minutes on modern hardware
- **Data Privacy**: Uses anonymized smart home data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository
- Framework: TensorFlow/Keras, scikit-learn
- Inspiration: Smart home energy optimization research

---

**⭐ Star this repository if you find it helpful!**

For questions or issues, please open a GitHub issue or contact the maintainers.

