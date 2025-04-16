ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y \
    python3.10 \
    python3-pip \
    build-essential \
    curl \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY ./src/rag_server/requirements.txt /opt/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/requirements.txt \
    && pip3 install --no-cache-dir typing-extensions>=4.8.0

# Create workspace and temp directories
WORKDIR /workspace
RUN mkdir -p /tmp-data/uploaded_files && \
    chmod 777 -R /tmp-data && \
    chown 1000:1000 -R /tmp-data

# Set environment variables
ENV PYTHONPATH=/workspace:/workspace/src
ENV HF_HOME=/tmp-data

# Copy application code
COPY ./ /workspace/

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

ENTRYPOINT ["uvicorn", "src.rag_server.server:app", "--host", "0.0.0.0", "--port", "8000"] 