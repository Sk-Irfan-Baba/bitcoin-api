# 1. Base Image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (for caching)
COPY requirements.txt .

# 4. Install dependencies
# We need to install system dependencies for some python libraries if needed
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY THE MODEL AND SCALER INTO THE CONTAINER
COPY bitcoin_model.h5 .
COPY scaler.pkl .
COPY main.py .

# 6. Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]