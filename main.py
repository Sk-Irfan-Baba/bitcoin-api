# main.py
from fastapi import FastAPI, HTTPException
import yfinance as yf
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Global variables to hold artifacts
model = None
scaler = None

@app.on_event("startup")
async def load_artifacts():
    global model, scaler
    try:
        # Load the model and scaler only once when the server starts
        model = tf.keras.models.load_model('bitcoin_model.h5')
        scaler = joblib.load('scaler.pkl')
        print("✅ Artifacts loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

@app.get("/")
def home():
    return {"message": "Bitcoin Predictor API is running!"}

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Fetch Data (Get extra days to be safe)
        data = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="Ticker not found")
        
        # Handle MultiIndex if necessary
        if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame): 
             prices = data['Close'][ticker].values
        else:
             prices = data['Close'].values

        # 2. Prepare Last 60 Days
        if len(prices) < 60:
            raise HTTPException(status_code=400, detail="Not enough data")
            
        last_60_days = prices[-60:].reshape(-1, 1)
        
        # 3. Scale
        scaled_data = scaler.transform(last_60_days)
        
        # 4. Reshape for LSTM
        X_input = np.array([scaled_data]) # Shape: (1, 60, 1)
        
        # 5. Predict
        prediction_scaled = model.predict(X_input)
        prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
        
        current_price = prices[-1]
        
        return {
            "ticker": ticker,
            "current_price": round(float(current_price), 2),
            "predicted_price": round(float(prediction_price), 2),
            "trend": "UP 🚀" if prediction_price > current_price else "DOWN 🔻"
        }

    except Exception as e:
        return {"error": str(e)}
    