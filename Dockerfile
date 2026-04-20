# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Copy and install requirements first (for faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ONLY the files we need for the API to run
COPY main.py .
COPY model.pkl .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]