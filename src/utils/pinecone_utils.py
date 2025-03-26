import os
import logging
from typing import Optional
from pinecone import Pinecone, ApiException as PineconeApiException

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
                host=os.getenv("PINECONE_HOST", "http://localhost:5080")
            )
        else:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise PineconeConnectionError("PINECONE_API_KEY environment variable not set")
            client = Pinecone(api_key=api_key)
            
        # Test connection
        client.list_indexes()
        return client
        
    except PineconeApiException as e:
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
        return pc.Index(index_name)
    except PineconeApiException as e:
        logger.error(f"Pinecone API error accessing index: {str(e)}")
        raise PineconeConnectionError(f"Failed to access Pinecone index: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error accessing Pinecone index: {str(e)}")
        raise PineconeConnectionError(f"Unexpected error: {str(e)}") 