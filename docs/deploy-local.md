# Deploying with Pinecone Local

This guide covers setting up the RAG Blueprint using Pinecone Local for development and testing.

## Prerequisites

- Python 3.12 or later
- Docker and Docker Compose
- NVIDIA GPU drivers (for GPU acceleration)

## Quick Start

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/rag.git
cd rag
```

2. Start the services using Docker Compose:

```bash
docker compose -f docker-compose.local.yaml up
```

This will start:
- Pinecone Local server
- Dense index service
- RAG server
- Other required services

## Configuration

No additional configuration is needed for Pinecone Local. The following are set automatically:
- API Key: "dummy-key-for-local" (not used)
- Host: http://pinecone:5080
- Index Name: "rag-index"

## Limitations

Be aware of these Pinecone Local limitations:
- Maximum 100,000 vectors per index
- Data is not persistent (cleared on restart)
- Bulk operations not supported
- Single machine deployment only

## Verifying the Setup

1. Check if services are running:

```bash
docker compose -f docker-compose.local.yaml ps
```

2. Test the Pinecone Local endpoint:

```bash
curl http://localhost:5080/health
```

3. Access the RAG UI:

```
http://localhost:8000
```

## Development Workflow

1. **Ingest Documents**:
   - Use the Knowledge Base tab in the RAG UI
   - Or use the `/documents` API endpoint

2. **Query the System**:
   - Use the Chat tab in the RAG UI
   - Or use the `/generate` API endpoint

3. **Monitor Operations**:
   - Check Docker logs for issues:

     ```bash
     docker compose -f docker-compose.local.yaml logs -f
     ```

## Troubleshooting

Common issues and solutions:

1. **Service Won't Start**:
   - Ensure ports 5080-5090 are available
   - Check Docker logs for errors

2. **Index Creation Fails**:
   - Verify dimension matches your embedding size
   - Check available memory (needs ~2GB minimum)

3. **Data Persistence**:
   - Remember data is lost on restart
   - Export important vectors if needed

## Next Steps

- [Customize Your Vector Database](vector-database.md)
- [Deploy to Production](deploy-cloud.md) 