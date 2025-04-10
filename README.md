# ![NVIDIA Logo](https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f) AI Blueprint: RAG

Use the following documentation to learn about the NVIDIA RAG Blueprint.

## Prerequisites

- Python 3.12 or later
- Docker and Docker Compose
- NVIDIA GPU drivers (for GPU acceleration)
- A Pinecone account (for production deployment)
- NVIDIA API key for NIM services and NV-Ingest

### Getting Required API Keys

1. **NVIDIA API Key**:
   - Visit the [NVIDIA AI Enterprise Portal](https://ngc.nvidia.com/setup/api-key)
   - Log in or create an account
   - Navigate to "Setup" > "API Key"
   - Generate a new API key
   - Set as `NVIDIA_API_KEY` in your `.env` file

2. **Pinecone API Key** (for production):
   - Create account at [Pinecone](https://www.pinecone.io/)
   - Navigate to API Keys in your dashboard
   - Create a new API key
   - Set as `PINECONE_API_KEY` in your `.env` file

## Environment Setup

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Update the following variables in `.env`:

   ```bash
   # Required for NVIDIA services
   NVIDIA_API_KEY=your-nvidia-api-key
   
   # Required for Pinecone cloud deployment
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_CLOUD=aws  # Cloud provider (aws, gcp, azure)
   PINECONE_REGION=us-east-1  # Region in your chosen cloud
   
   # Vector store configuration
   PINECONE_INDEX_NAME=your-index-name    # Default: rag-index
   PINECONE_DIMENSION=1024                # Must match embedding dimension
   PINECONE_METRIC=cosine                 # Distance metric for vectors
   
   # Retrieval settings
   APP_RETRIEVER_TOPK=4                   # Number of documents to retrieve
   
   # Local development (optional)
   PINECONE_HOST=http://localhost:5080    # Only needed for local deployment
   ```

3. Authenticate with NVIDIA Container Registry:

   ```bash
   # Log in using your NVIDIA API key
   docker login nvcr.io -u '$oauthtoken' -p $NVIDIA_API_KEY
   ```

4. For local development, only `NVIDIA_API_KEY` is required. Other variables will use defaults.

5. For cloud deployment, ensure `PINECONE_API_KEY` and `PINECONE_REGION` are set correctly.

- [Overview](#overview)
- [Key Features](#key-features)
- [Target Audience](#target-audience)
- [Software Components](#software-components)
- [Technical Diagram](#technical-diagram)
- [Minimum System Requirements](#minimum-system-requirements)
  - [OS Requirements](#os-requirements)
  - [Deployment Options](#deployment-options)
  - [Driver versions](#driver-versions)
  - [Minimum hardware requirements for self hosting all NVIDIA NIM microservices](#minimum-hardware-requirements-for-self-hosting-all-nvidia-nim-microservices)
- [Next Steps](#next-steps)
- [Available Customizations](#available-customizations)
- [Vector Store Options](#vector-store-options)
- [Inviting the community to contribute](#inviting-the-community-to-contribute)
- [License](#license)
- [Deployment Options](#deployment-options)

## Overview

This blueprint serves as a reference solution for a foundational Retrieval Augmented Generation (RAG) pipeline.
One of the key use cases in Generative AI is enabling users to ask questions and receive answers based on their enterprise data corpus.
This blueprint demonstrates how to set up a RAG solution that uses NVIDIA NIM and GPU-accelerated components.
By default, this blueprint leverages locally-deployed NVIDIA NIM microservices to meet specific data governance and latency requirements.
However, you can replace these models with your NVIDIA-hosted models available in the [NVIDIA API Catalog](https://build.nvidia.com).

## Key Features
- Multimodal data extraction support with text, tables, charts, and infographics
- Hybrid search with dense and sparse search
- Multilingual and cross-lingual retrieval
- Reranking to further improve accuracy
- GPU-accelerated Index creation and search
- Multi-turn conversations. Opt-in query rewriting for better accuracy.
- Multi-session support
- Telemetry and observability
- Opt-in for query rewriting to improve multiturn accuracy
- Opt-in for reflection to improve accuracy
- Opt-in for guardrailing conversations
- Opt-in image captioning with vision language models (VLMs)
- Sample user interface
- OpenAI-compatible APIs
- Decomposable and customizable

## Target Audience

This blueprint is for:

- **Developers**: Developers who want a quick start to set up a RAG solution for unstructured data with a path-to-production with the NVIDIA NIM.

## Software Components

The following are the default components included in this blueprint:

- NVIDIA NIM Microservices
  - Response Generation (Inference)
    - [NIM of meta/llama-3.1-70b-instruct](https://build.nvidia.com/meta/llama-3_1-70b-instruct)
  - Retriever Models
    - [NIM of nvidia/llama-3_2-nv-embedqa-1b-v2]( https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2)
    - [NIM of nvidia/llama-3_2-nv-rerankqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2)
- NV-Ingest Client Library
  - Document parsing and chunking
  - Multi-modal support (text, tables, charts, images)
  - Page-level extraction
- Orchestrator server - NV-Ingest based
- Pinecone Vector Database
  - Cloud service for production deployment
  - Local development server for testing
- File Types: [File types supported](https://docs.unstructured.io/platform/supported-file-types) by unstructured.io. Accuracy is best optimized for files with extension `.pdf`, `.txt` and `.md`.

We provide Docker Compose scripts that deploy the microservices on a single node.
When you are ready for a large-scale deployment,
you can use the included Helm charts to deploy the necessary microservices.
You use sample Jupyter notebooks with the JupyterLab service to interact with the code directly.

The Blueprint contains sample data from the [NVIDIA Developer Blog](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/data/dataset.zip) and also some [sample multimodal data](./data/multimodal/).
You can build on this blueprint by customizing the RAG application to your specific use case.

We also provide a sample user interface named `rag-playground`.

## Technical Diagram

![RAG Architecture Diagram](./docs/arch_diagram.png)

The image represents the high level architecture and workflow. The core business logic uses NVIDIA's NV-Ingest pipeline for document processing and retrieval. Here's a step-by-step explanation of the workflow from end-user perspective:

1. **User Interaction via RAG Playground**:
   - The user interacts with this blueprint by typing queries into the sample UI microservice named as **RAG Playground**. These queries are sent to the system through the `POST /generate` API exposed by the RAG server microservice.

2. **Query Processing**:
   - The query enters the **RAG Server**, which uses NVIDIA's NV-Ingest pipeline for document processing and retrieval.

3. **Retrieval of Relevant Documents**:
   - The refined query is passed to the **Retriever** module. This component queries the **Pinecone Vector Database service**, which stores embeddings of unstructured data, generated using **NeMo Retriever Embedding microservice**. The retriever module identifies the top 20 most relevant chunks of information related to the query.

4. **Reranking for Precision**:
   - The top K chunks are passed to the optional **NeMo Retriever reranking microservice**. The reranker narrows down the results to the top N most relevant chunks, improving precision.

5. **Response Generation**:
   - The top N chunks are injected in the prompt and sent to the **Response Generation** module, which leverages **NeMo LLM inference Microservice** to generate a natural language response based on the retrieved information. Optionally, a reflection module can be enabled which makes additional LLM calls to improve the response by verifying its groundness based on retrieved context at this stage. NeMo guardrails can also be enabled at this stage to guardrail the output against toxicity.

6. **Delivery of Response**:
   - The generated response is sent back to the **RAG Playground**, where the user can view the answer to their query as well as check the output of the retriever module using the `Citations` option.

7. **Ingestion of Data**:
   - Separately, unstructured data is ingested into the system via the `POST /documents` API using the `Knowledge Base` tab of **RAG Playground microservice**. This data is preprocessed, split into chunks and stored in the **Pinecone Vector Database** using embeddings generated by models hosted by **NeMo Retriever Embedding microservice**.

This modular design ensures efficient query processing, accurate retrieval of information, and easy customization.

## Hardware Requirements

Following are the hardware requirements for each component.
The reference code in the solution (glue code) is referred to as as the "pipeline".

The overall hardware requirements depend on whether you
[Deploy With Docker Compose](/docs/quickstart.md#deploy-with-docker-compose) or [Deploy With Helm Chart](/docs/quickstart.md#deploy-with-helm-chart).

### Driver versions

- Python 3.12 or later
- GPU Driver -  530.30.02 or later
- CUDA version - 12.6 or later

### Minimum hardware requirements for self hosting all NVIDIA NIM microservices

**The NIM and hardware requirements only need to be met if you are self-hosting them with default settings of RAG.**
See [Using self-hosted NVIDIA NIM microservices](./docs/quickstart.md#deploy-with-docker-compose).

- **Pipeline operation**: No local GPU required for vector database operations as Pinecone is cloud-hosted
- (If locally deployed) **LLM NIM**: [Meta Llama 3.1 70B Instruct Support Matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html#llama-3-1-70b-instruct)
  - For improved paralleled performance, we recommend 8x or more H100s for LLM inference.
  - The pipeline can share the GPU with the LLM NIM, but it is recommended to have a separate GPU for the LLM NIM for optimal performance.
- **Embedding NIM**: [Llama-3.2-NV-EmbedQA-1B-v2 Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html#llama-3-2-nv-embedqa-1b-v2)
  - The pipeline can share the GPU with the Embedding NIM, but it is recommended to have a separate GPU for the Embedding NIM for optimal performance.
- (If locally deployed) **Reranking NIM**: [llama-3_2-nv-rerankqa-1b-v1 Support Matrix](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html#llama-3-2-nv-rerankqa-1b-v2)

## Next Steps

- Do the procedures in [Get Started](/docs/quickstart.md) to deploy this blueprint
- See the [OpenAPI Specification](/docs/api_reference/openapi_schema.json)
- Explore notebooks that demonstrate how to use the APIs [here](/notebooks/)

## Available Customizations

The following are some of the customizations that you can make after you complete the steps in [Get Started](/docs/quickstart.md).

- [Change the Inference or Embedding Model](docs/change-model.md)
- [Customize Prompts](docs/prompt-customization.md)
- [Customize LLM Parameters at Runtime](docs/llm-params.md)
- [Support Multi-Turn Conversations](docs/multiturn.md)

## Inviting the community to contribute

We're posting these examples on GitHub to support the NVIDIA LLM community and facilitate feedback.
We invite contributions!
To open a GitHub issue or pull request, see the [contributing guidelines](./CONTRIBUTING.md).

## License

This NVIDIA NVIDIA AI BLUEPRINT is licensed under the [Apache License, Version 2.0.](./LICENSE) This project will download and install additional third-party open source software projects and containers. Review [the license terms of these open source projects](./LICENSE-3rd-party.txt) before use.

The software and materials are governed by the NVIDIA Software License Agreement (found at <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/>) and the Product-Specific Terms for NVIDIA AI Products (found at <https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/>), except that models are governed by the AI Foundation Models Community License Agreement (found at NVIDIA Agreements | Enterprise Software | NVIDIA Community Model License) and NVIDIA dataset is governed by the NVIDIA Asset License Agreement found [here](./data/LICENSE.DATA).

For Meta/llama-3.1-70b-instruct model the Llama 3.1 Community License Agreement, for nvidia/llama-3.2-nv-embedqa-1b-v2model the Llama 3.2 Community License Agreement, and for the nvidia/llama-3.2-nv-rerankqa-1b-v2 model the Llama 3.2 Community License Agreement. Built with Llama.

## Deployment Options

This blueprint supports two deployment options for the vector database:

### 1. Pinecone Cloud Service

Uses Pinecone's managed cloud service for production deployments. This option:

- Provides persistent storage
- Supports larger datasets
- Offers better scalability and reliability
- Requires a Pinecone account and API key

### 2. Pinecone Local (Development)

Uses Pinecone's local development server. This option:

- Runs entirely on your local machine
- Suitable for development and testing
- Limited to 100,000 records
- Data is not persistent (in-memory only)
- No authentication required

Choose the appropriate deployment option based on your needs:

- For production use → [Deploy with Pinecone Cloud](/docs/deploy-cloud.md)
- For local development → [Deploy with Pinecone Local](/docs/deploy-local.md)

## Vector Store Options

This blueprint supports two deployment options for vector storage:

### 1. Pinecone Cloud Service (Production)

Uses Pinecone's managed cloud service, recommended for production deployments. This option:

- Provides persistent storage across restarts
- Supports large-scale datasets
- Offers better scalability and reliability
- Requires a Pinecone account and API key
- Available in multiple regions and cloud providers

To use Pinecone Cloud:

```bash
# Set your Pinecone credentials
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="your-environment"  # e.g., "gcp-starter"

# Deploy with cloud configuration
docker compose -f docker-compose.cloud.yaml up
```

### 2. Pinecone Local (Development)

Uses Pinecone's local development server, ideal for testing and development. This option:

- Runs entirely on your local machine
- No authentication required
- Perfect for development and testing
- **Limitations**:
  - Maximum 100,000 vectors per index
  - Data is not persistent (in-memory only)
  - Some features like bulk import not supported
  - Limited to single machine deployment

To use Pinecone Local:

```bash
# Deploy with local configuration
docker compose -f docker-compose.local.yaml up
```

### Switching Between Environments

You can maintain both configurations and switch between them:

```bash
# For local development
docker compose -f docker-compose.local.yaml up

# For production
docker compose -f docker-compose.cloud.yaml up
```

The application will automatically detect which environment it's running in and configure itself appropriately.

### Environment Variables

#### For Pinecone Cloud

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX_NAME`: Name of your index

#### For Pinecone Local

- No configuration needed - runs out of the box
- Default index name: "rag-index"
- Local endpoint: <http://localhost:5080>

#### Required Variables

- `NVIDIA_API_KEY`: Your NVIDIA API key for NIM services

#### Optional Variables

- `PINECONE_API_KEY`: Your Pinecone API key (required for cloud deployment)
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (for cloud deployment)
- `PINECONE_INDEX_NAME`: Name of your index
- `NV_INGEST_HOST`: Host for Nemo Retriever Extract service

## Document Processing Pipeline

This blueprint uses NVIDIA's Nemo Retriever Extract (formerly NV-Ingest) for document processing:

1. **Document Ingestion**:
   - Supports multiple file formats including PDFs, Word docs, and PowerPoint
   - Extracts text, tables, charts, and images
   - Processes at page-level granularity

2. **Vector Storage**:
   - Uses Pinecone for vector storage and retrieval
   - Supports both cloud and local deployment
   - Automatic index creation and management

3. **Retrieval**:
   - Semantic search using NVIDIA's embedding models
   - Reranking for improved accuracy
   - Multi-modal context support

## Troubleshooting

### Pinecone Issues

1. **Connection Errors**:

   ```
   PineconeConnectionError: Failed to connect to Pinecone
   ```

   - Check if PINECONE_API_KEY is set correctly
   - Verify PINECONE_HOST for local deployment
   - Ensure PINECONE_ENVIRONMENT is correct for cloud deployment

2. **Index Creation Fails**:

   ```
   PineconeApiException: Index creation failed
   ```

   - Verify dimension matches your embedding model (default: 1536)
   - Check if index name is unique
   - Ensure you have permissions to create indexes

3. **Query Issues**:

   ```
   PineconeApiException: Query failed
   ```

   - Verify vector dimensions match index configuration
   - Check if index is populated with vectors
   - Ensure top_k value is within limits

### Nemo Retriever Extract Issues

1. **Service Connection**:

   ```
   ConnectionError: Failed to connect to NV-Ingest service
   ```

   - Verify NV_INGEST_HOST is correct
   - Check if service is running (`docker compose ps`)

2. **Document Processing**:

   ```
   Error processing documents: Extraction failed
   ```

   - Check file format compatibility
   - Verify file permissions
   - Ensure sufficient system resources

3. **Embedding Generation**:

   ```
   NVIDIAServiceError: Embedding generation failed
   ```

   - Verify NVIDIA_API_KEY is valid
   - Check model availability
   - Monitor GPU resource usage

### Docker Issues

1. **Container Startup**:

   ```
   Error starting userland proxy: listen tcp4 0.0.0.0:5080: bind: address already in use
   ```

   - Check for port conflicts (5080-5090, 7671, 8000)
   - Stop any running containers using same ports
   - Verify Docker service is running

2. **Resource Issues**:

   ```
   Container exited with code 137
   ```

   - Increase Docker memory limits
   - Monitor system resources
   - Check container logs for OOM errors

### Common Solutions

1. **Reset Local Environment**:

   ```bash
   # Stop and remove containers
   docker compose -f docker-compose.local.yaml down

   # Remove volumes
   docker volume prune -f

   # Rebuild and restart
   docker compose -f docker-compose.local.yaml up --build
   ```

2. **Check Service Health**:

   ```bash
   # Check Pinecone
   curl http://localhost:5080/health

   # Check NV-Ingest
   curl http://localhost:7671/health

   # Check RAG server
   curl http://localhost:8000/health
   ```

3. **View Logs**:

   ```bash
   # All services
   docker compose -f docker-compose.local.yaml logs -f

   # Specific service
   docker compose -f docker-compose.local.yaml logs -f rag-server
   ```

For additional help, check the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) or [Pinecone Documentation](https://docs.pinecone.io/).
