import os
import logging
from typing import Optional
from pinecone import Pinecone, PineconeException, ServerlessSpec

logger = logging.getLogger(__name__)

class PineconeConnectionError(Exception):
    """Custom exception for Pinecone connection issues"""
    pass

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