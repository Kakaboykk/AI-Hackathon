#!/usr/bin/env python3
"""
Minimal script to run the appliances energy prediction notebook without matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Starting Appliances Energy Prediction using LSTMs...")
print("=" * 50)

# Loading the dataset
print("\n1. Loading the dataset...")
df = pd.read_csv('energydata_complete.csv', index_col=0, parse_dates=True)
print(f"Dataset shape: {df.shape}")
print(f"Dataset info:")
print(df.info())

# Exploring the dataset
print("\n2. Exploring the dataset...")
print(f"Dataset shape: {df.shape}")

# Round the data
df = df.round(2)
print(f"First few rows:")
print(df.head())

# Splitting the dataset
print("\n3. Splitting the dataset...")
test_size = 30 * 144  # 30 days / 1 month
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature Scaling
print("\n4. Feature Scaling...")
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)
print(f"Scaled train shape: {scaled_train.shape}")
print(f"Scaled test shape: {scaled_test.shape}")

# Generating time series batches
print("\n5. Creating time series generators...")
length = 288
batch_size = 16

train_generator = TimeseriesGenerator(data=scaled_train, targets=scaled_train, length=length, batch_size=batch_size)
validation_generator = TimeseriesGenerator(data=scaled_test, targets=scaled_test, length=length, batch_size=batch_size)

print(f"Train generator length: {len(train_generator)}")
print(f"Validation generator length: {len(validation_generator)}")

# Model Training & Evaluation
print("\n6. Building and training the model...")
model = Sequential()
model.add(LSTM(units=150, activation='relu', input_shape=(length, scaled_train.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(units=scaled_train.shape[1]))

print("Model summary:")
model.summary()

model.compile(loss='mse', optimizer='adam')

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True, start_from_epoch=6, verbose=1)

# Training
print("\n7. Training the model...")
history = model.fit(
    x=train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[es],
    verbose=1
)

# Print training losses
print("\n8. Training losses:")
losses = pd.DataFrame(history.history)
print("Final training loss:", losses['loss'].iloc[-1])
print("Final validation loss:", losses['val_loss'].iloc[-1])

# Making predictions
print("\n9. Making predictions on test set...")
test_predictions = []
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))

for i in tqdm(range(len(test)), desc="Generating predictions"):
    pred = model.predict(current_batch, verbose=0)[0]
    test_predictions.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

# Converting predictions back to original scale
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions = pd.DataFrame(true_predictions, columns=test.columns)

# Calculate RMSE scores
print("\n10. Calculating evaluation metrics...")
rmse_scores = []
for col in test.columns:
    rmse_scores.append(np.sqrt(mean_squared_error(test[col], true_predictions[col])))

print(f"Mean RMSE Score: {round(np.mean(rmse_scores), 5)}")

# Show some sample predictions
print("\nSample predictions for Appliances:")
print(f"Actual: {test['Appliances'].head().values}")
print(f"Predicted: {true_predictions['Appliances'].head().values}")

# Re-training on full dataset for forecasting
print("\n11. Re-training model on full dataset for forecasting...")
full_scaler = MinMaxScaler()
scaled_df = full_scaler.fit_transform(df)

generator = TimeseriesGenerator(data=scaled_df, targets=scaled_df, length=length, batch_size=batch_size)

# New model with more units
model_full = Sequential()
model_full.add(LSTM(units=200, activation='relu', input_shape=(length, scaled_df.shape[1])))
model_full.add(Dropout(0.3))
model_full.add(Dense(units=scaled_df.shape[1]))

model_full.compile(loss='mse', optimizer='adam')

print("Training final model on full dataset...")
history_full = model_full.fit(x=generator, epochs=10, verbose=1)

# Print final training losses
print("\n12. Final training losses:")
losses_full = pd.DataFrame(history_full.history)
print("Final training loss:", losses_full['loss'].iloc[-1])

# Generate forecasts
print("\n13. Generating future forecasts...")
forecast = []
first_eval_batch = scaled_df[-length:]
current_batch = first_eval_batch.reshape(1, length, scaled_df.shape[1])

for i in tqdm(range(len(test)), desc="Generating forecasts"):
    pred = model_full.predict(current_batch, verbose=0)[0]
    forecast.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

forecast = np.array(forecast)
true_forecast = full_scaler.inverse_transform(forecast)

# Create forecast index
forecast_index = pd.date_range(start='27-05-2016 18:10', periods=30*144, freq='10min')
true_forecast_df = pd.DataFrame(true_forecast, index=forecast_index, columns=df.columns)

# Show forecast statistics
print("\n14. Forecast statistics:")
print(f"Appliances forecast - Mean: {true_forecast_df['Appliances'].mean():.2f}, Std: {true_forecast_df['Appliances'].std():.2f}")
print(f"Windspeed forecast - Mean: {true_forecast_df['Windspeed'].mean():.2f}, Std: {true_forecast_df['Windspeed'].std():.2f}")

# Save the model
print("\n15. Saving the trained model...")
model_full.save('appliances_energy_predictor.keras')
print("Model saved as 'appliances_energy_predictor.keras'")

# Save forecast data
print("\n16. Saving forecast data...")
true_forecast_df.to_csv('forecast_data.csv')
print("Forecast data saved as 'forecast_data.csv'")

print("\n" + "=" * 50)
print("Notebook execution completed successfully!")
print("Generated files:")
print("- appliances_energy_predictor.keras (trained model)")
print("- forecast_data.csv (forecast results)")
print("=" * 50)
