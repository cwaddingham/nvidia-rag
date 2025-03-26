# ![NVIDIA Logo](https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f) AI Blueprint: RAG

Use the following documentation to learn about the NVIDIA RAG Blueprint.

## Prerequisites

- Python 3.12 or later
- Docker and Docker Compose
- NVIDIA GPU drivers (for GPU acceleration)
- A Pinecone account (for production deployment)

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
- Orchestrator server - Langchain based
- Pinecone Vector Database - cloud-hosted vector database service
- Text Splitter: [Recursive Character Text Splitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
- Document parsers: [Unstructured.io](https://docs.unstructured.io)
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

The image represents the high level architecture and workflow. The core business logic is defined in the `rag_chain_with_multiturn()` method of `chains.py` file. Here's a step-by-step explanation of the workflow from end-user perspective:

1. **User Interaction via RAG Playground**:
   - The user interacts with this blueprint by typing queries into the sample UI microservice named as **RAG Playground**. These queries are sent to the system through the `POST /generate` API exposed by the RAG server microservice. There are separate [notebooks](./notebooks/) available which showcase API usage as well.

2. **Query Processing**:
   - The query enters the **RAG Server**, which is based on LangChain. An optional **Query Rewriter** component may refine or decontextualize the query for better retrieval results at this stage. An optional NeMoGuardrails component can be enabled as well to help filter out queries at input of the pipeline.

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
