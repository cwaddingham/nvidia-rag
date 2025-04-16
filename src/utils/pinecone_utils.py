import os
import logging
from typing import Optional, List, Any, Dict
from pinecone import Pinecone, PineconeException, ServerlessSpec
from src.ingestor_server.document import Document

logger = logging.getLogger(__name__)

class PineconeConnectionError(Exception):
    """Custom exception for Pinecone connection issues"""
    pass

class PineconeEmbedder: 
    """Pinecone embedding model wrapper"""
    def __init__(self, model_name: str, input_type: str = "query", truncate: str = "END"):
        self.model_name = model_name
        self.input_type = input_type
        self.truncate = truncate
        self.client = get_pinecone_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Pinecone"""
        return self.client.embed(texts, input_type=self.input_type, truncate=self.truncate)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Pinecone"""
        return self.client.embed(text, input_type=self.input_type, truncate=self.truncate)

def get_pinecone_client() -> Pinecone:
    """Get initialized Pinecone client based on environment"""
    try:
        is_local = os.getenv("PINECONE_HOST", "").startswith("http://")
        
        if is_local:
            client = Pinecone(
                api_key="pclocal",
                host=os.getenv("PINECONE_HOST", "http://localhost:5081")
            )
        else:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise PineconeConnectionError("PINECONE_API_KEY environment variable not set")
            client = Pinecone(
                api_key=api_key,
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
            
        # Test connection
        client.list_indexes()
        return client
        
    except PineconeException as e:
        logger.error(f"Pinecone API error: {str(e)}")
        raise PineconeConnectionError(f"Failed to connect to Pinecone: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error connecting to Pinecone: {str(e)}")
        raise PineconeConnectionError(f"Unexpected error: {str(e)}")

def get_index(pc: Optional[Pinecone] = None):
    """Get Pinecone index from client"""
    try:
        if pc is None:
            pc = get_pinecone_client()
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
        if not index_name in pc.list_indexes():
            create_index(pc, 
                         index_name, 
                         os.getenv("PINECONE_DIMENSION", 1536), 
                         os.getenv("PINECONE_METRIC", "cosine"), 
                         os.getenv("PINECONE_SERVERLESS", True), 
                         os.getenv("PINECONE_CLOUD", "aws"), 
                         os.getenv("PINECONE_REGION", "us-east-1"))
        return pc.Index(index_name)
    except PineconeException as e:
        logger.error(f"Pinecone API error accessing index: {str(e)}")
        raise PineconeConnectionError(f"Failed to access Pinecone index: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error accessing Pinecone index: {str(e)}")
        raise PineconeConnectionError(f"Unexpected error: {str(e)}") 

def list_indexes(pc: Pinecone) -> List[str]:
    """List all indexes in Pinecone"""
    try:
        return pc.list_indexes()
    except PineconeException as e:
        logger.error(f"Pinecone API error listing indexes: {str(e)}")
        raise PineconeConnectionError(f"Failed to list Pinecone indexes: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error listing Pinecone indexes: {str(e)}")

def create_index(pc: Pinecone, index_name: str, dimension: int, metric: str = "cosine", serverless: bool = True, cloud: str = "aws", region: str = "us-east-1"):
    """Create Pinecone index if it doesn't exist"""
    pc.create_index(
        index_name, 
        dimension=dimension, 
        metric=metric, 
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        ) if serverless else None
    )
    logger.info(f"Index {index_name} created successfully")

def delete_index(pc: Pinecone, index_name: str) -> None:
    """Delete Pinecone index"""
    try:
        pc.delete_index(index_name)
        logger.info(f"Index {index_name} deleted successfully")
    except PineconeException as e:
        logger.error(f"Pinecone API error deleting index: {str(e)}")
        raise PineconeConnectionError(f"Failed to delete Pinecone index: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error deleting Pinecone index: {str(e)}")
        raise PineconeConnectionError(f"Unexpected error: {str(e)}")

def add_documents(pc: Pinecone, index_name: str, documents: List[Document], embedder: Any, batch_size: int = 500) -> None:
    """Add documents to Pinecone index with embeddings"""
    index = pc.Index(index_name)
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Generate embeddings for the batch
        texts = [doc.content for doc in batch]
        embeddings = embedder.embed_documents(texts)
        
        # Prepare vectors for Pinecone
        vectors = [
            {
                'id': doc.metadata['document_id'],
                'values': embedding,
                'metadata': {
                    'text': doc.content,
                    'filename': doc.metadata['filename'],
                    'page': doc.metadata.get('page', 0),
                    'chunk': doc.metadata.get('chunk', 0),
                    'source': doc.metadata.get('source', '')
                }
            }
            for doc, embedding in zip(batch, embeddings)
        ]
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)

def get_index_stats(pc: Pinecone, index_name: str) -> Dict[str, Any]:
    """Get index statistics from Pinecone"""
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "metric": stats.metric,
            "dimension": stats.dimension,
            "namespaces": list(stats.namespaces.keys()) if stats.namespaces else [''],
            "vector_type": stats.vector_type,
        }
    except Exception as e:
        logger.error(f"Failed to get Pinecone index stats: {str(e)}")
        raise