# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 