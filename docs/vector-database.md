<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Customize Your Vector Database

By default, the Docker Compose files for the examples deploy [Milvus](https://milvus.io/) as the vector database with CPU-only support.
You must install the NVIDIA Container Toolkit to use Milvus with GPU acceleration. 
You can also extend the code to add support for any vector store. 
This blueprint uses [Pinecone](https://www.pinecone.io/) as the vector database. You can choose between:

- Pinecone Cloud for production deployments
- Pinecone Local for development and testing

- [Configure Pinecone Cloud](#configure-pinecone-cloud)
- [Configure Pinecone Local](#configure-pinecone-local)
- [Vector Store Configuration](#vector-store-configuration)

## Configure Pinecone Cloud

For production deployments, use Pinecone's cloud service:

1. Create a Pinecone account and get your API key
2. Set environment variables in `.env`:

```bash
PINECONE_API_KEY=your-api-key
PINECONE_CLOUD=aws  # Cloud provider (aws, gcp, azure)
PINECONE_REGION=us-east-1  # Region in your chosen cloud
PINECONE_INDEX_NAME=your-index-name
```

3. Start services with cloud configuration:

```bash
docker compose -f docker-compose.cloud.yaml up
```

## Configure Pinecone Local

For development and testing, use Pinecone Local:

1. No configuration needed - runs out of the box
2. Start services with local configuration:

```bash
docker compose -f docker-compose.local.yaml up
```

Note: Pinecone Local has some limitations:

- Maximum 100,000 vectors per index
- Data is not persistent (cleared on restart)
- Single machine deployment only

## Vector Store Configuration

Common configuration options:

- `PINECONE_INDEX_NAME`: Name of your index (default: rag-index)
- `PINECONE_DIMENSION`: Vector dimension (default: 1536)
- `PINECONE_METRIC`: Distance metric (default: cosine)

For cloud deployment:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_CLOUD`: Your cloud provider (aws, gcp, azure)
- `PINECONE_REGION`: Region in your chosen cloud (e.g., us-east-1)

For local development:

- `PINECONE_HOST`: Local endpoint (default: http://localhost:5080)
