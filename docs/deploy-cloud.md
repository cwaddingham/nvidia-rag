# Deploying with Pinecone Cloud

This guide covers deploying the RAG Blueprint using Pinecone's cloud service for production use.

## Prerequisites

- Python 3.12 or later
- Docker and Docker Compose
- NVIDIA GPU drivers (for GPU acceleration)
- Pinecone account with API key
- Active Pinecone cloud account

## Setup

1. Clone the repository:

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/rag.git
cd rag
```

2. Set up environment variables:

```bash
# Create .env file
cat << EOF > .env
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=your-index-name
EOF
```

3. Start the services:

```bash
docker compose -f docker-compose.cloud.yaml up
```

## Configuration Options

### Required Environment Variables

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_CLOUD`: Your cloud provider (aws, gcp, azure)
- `PINECONE_REGION`: Region in your chosen cloud (e.g., us-east-1)
- `PINECONE_INDEX_NAME`: Name for your index

### Optional Configuration

- `PINECONE_DIMENSION`: Vector dimension (default: 1536)
- `PINECONE_METRIC`: Distance metric (default: "cosine")

## Production Considerations

1. **Index Management**:
   - Create indexes in advance
   - Monitor index usage and metrics
   - Set up proper access controls

2. **Performance Optimization**:
   - Choose appropriate pod type
   - Consider index replicas for high availability
   - Monitor query latency

3. **Security**:
   - Secure API keys
   - Use environment-specific configurations
   - Enable API request logging

## Monitoring and Maintenance

1. **Health Checks**:

```bash
# Check RAG server status
curl http://localhost:8000/health

# Monitor Docker containers
docker compose -f docker-compose.cloud.yaml ps
```

2. **Logging**:

```bash
# View service logs
docker compose -f docker-compose.cloud.yaml logs -f
```

3. **Backup and Recovery**:
   - Regular index backups recommended
   - Document source tracking
   - Version control for configurations

## Scaling Guidelines

1. **Vector Database**:
   - Monitor index size and performance
   - Scale pod size as needed
   - Consider multiple indexes for large datasets

2. **Application**:
   - Adjust batch sizes for ingestion
   - Configure connection pooling
   - Set appropriate timeouts

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**:
   - Verify API key and environment
   - Check network connectivity
   - Confirm index exists

2. **Performance Problems**:
   - Monitor query latency
   - Check index statistics
   - Verify resource allocation

3. **Data Management**:
   - Track ingestion status
   - Monitor index size
   - Check for failed operations

## Security Best Practices

1. **API Key Management**:
   - Use environment variables
   - Rotate keys regularly
   - Implement least privilege access

2. **Network Security**:
   - Use secure connections
   - Implement rate limiting
   - Monitor access patterns

## Next Steps

- [Customize Your Vector Database](vector-database.md)
- [Performance Tuning](performance-tuning.md)
- [Local Development Setup](deploy-local.md) 