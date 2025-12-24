FROM python:3.12.7-slim

WORKDIR /app

# Install dependencies first
COPY conda.yaml requirements.txt* ./
RUN pip install --no-cache-dir pandas==2.1.4 numpy==1.26.2 scikit-learn==1.3.2 mlflow==2.19.0 setuptools

# Copy project files
COPY . .

# Default command
CMD ["python", "modelling.py"]
