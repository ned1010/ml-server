# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Run the application with dynamic port (no reload in production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
