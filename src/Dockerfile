# Base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir typing-extensions>=4.8.0

# Copy application code
COPY . .

# Environment variables for Pinecone Local
ENV PINECONE_API_KEY="dummy-key-for-local"
ENV PINECONE_HOST="http://pinecone:5080"
ENV PINECONE_INDEX_NAME="rag-index"
ENV NV_INGEST_HOST="http://nv-ingest:7671"
ENV NVIDIA_API_KEY=${NVIDIA_API_KEY}
ENV PYTHONPATH=/app/src

# Create directory for uploaded files
RUN mkdir -p /tmp-data/uploaded_files
RUN chmod 777 -R /tmp-data

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
