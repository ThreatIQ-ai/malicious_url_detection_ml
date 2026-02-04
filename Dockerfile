# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional but safe for numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for Docker layer caching)
COPY flask_req.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r flask_req.txt

# Copy the entire project
COPY . .

# Expose Flask port
EXPOSE 8000

# Run the Flask app
CMD ["python", "app.py"]
