import os

# Service URLs and ports configuration
RAG_SERVER_HOST = os.getenv("RAG_SERVER_HOST", "localhost")
RAG_SERVER_PORT = int(os.getenv("RAG_SERVER_PORT", "8000"))

INGESTOR_SERVER_HOST = os.getenv("INGESTOR_SERVER_HOST", "localhost")
INGESTOR_SERVER_PORT = int(os.getenv("INGESTOR_SERVER_PORT", "8080"))

MINIO_HOST = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT = int(os.getenv("MINIO_PORT", "9010"))

PINECONE_HOST = os.getenv("PINECONE_HOST", "localhost")
PINECONE_PORT = int(os.getenv("PINECONE_PORT", "5080"))

def get_service_url(host: str, port: int) -> str:
    """Generate service URL from host and port"""
    return f"http://{host}:{port}"

# Service URL getters
def get_rag_server_url() -> str:
    return get_service_url(RAG_SERVER_HOST, RAG_SERVER_PORT)

def get_ingestor_server_url() -> str:
    return get_service_url(INGESTOR_SERVER_HOST, INGESTOR_SERVER_PORT)

def get_minio_url() -> str:
    return get_service_url(MINIO_HOST, MINIO_PORT)

def get_pinecone_url() -> str:
    return get_service_url(PINECONE_HOST, PINECONE_PORT) 