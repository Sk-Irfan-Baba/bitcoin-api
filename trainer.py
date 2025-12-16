import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# --- 1. FETCH DATA ---
ticker = "BTC-USD"
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)

if data.empty:
    print("❌ Error: No data found.")
    exit()

# Handle MultiIndex if present (fix for yfinance update)
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Close'][ticker].values.reshape(-1, 1)
else:
    prices = data['Close'].values.reshape(-1, 1)

# --- 2. PREPARE DATA ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences
prediction_days = 60
x_data, y_data = [], []

for i in range(prediction_days, len(scaled_prices)):
    x_data.append(scaled_prices[i-prediction_days:i, 0])
    y_data.append(scaled_prices[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# --- 3. SPLIT TRAIN / TEST ---
# We use 80% for training and 20% for testing to calculate accuracy
split_idx = int(len(x_data) * 0.8)

x_train = x_data[:split_idx]
y_train = y_data[:split_idx]
x_test = x_data[split_idx:]
y_test = y_data[split_idx:]

# --- 4. BUILD & TRAIN MODEL ---
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Training Model...")
model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

# --- 5. CALCULATE METRICS (The New Part) ---
print("\nCalculating Accuracy Metrics...")
predictions = model.predict(x_test)

# Invert scaling to get real prices (e.g., $95,000 instead of 0.8)
real_predictions = scaler.inverse_transform(predictions)
real_actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# A. Mean Absolute Error (MAE)
mae = mean_absolute_error(real_actuals, real_predictions)

# B. Directional Accuracy (Did we predict UP/DOWN correctly?)
# We compare the change from the PREVIOUS day
actual_change = np.diff(real_actuals.flatten())
pred_change = np.diff(real_predictions.flatten())

# Check if the sign is the same (Both + or Both -)
correct_direction = np.sign(actual_change) == np.sign(pred_change)
direction_accuracy = np.mean(correct_direction) * 100

print(f"\n📊 --- MODEL PERFORMANCE ---")
print(f"💰 Average Error (MAE): ${mae:.2f}")
print(f"📈 Directional Accuracy: {direction_accuracy:.2f}%")
print(f"---------------------------")

# --- 6. SAVE ---
model.save("bitcoin_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("Model saved successfully.")