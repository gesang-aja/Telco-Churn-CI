# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install pandas==2.1.4 numpy==1.26.2 scikit-learn==1.3.2 mlflow==2.19.0

# Default command
CMD ["python", "modelling.py"]
