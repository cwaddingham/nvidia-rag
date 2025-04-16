from typing import List, Dict, Any
import os
import logging
from openai import OpenAI, OpenAIError
from ..utils.pinecone_utils import get_index, PineconeConnectionError

logger = logging.getLogger(__name__)

class NVIDIAServiceError(Exception):
    """Custom exception for NVIDIA API service issues"""
    pass

class NVIDIARetriever:
    def __init__(self):
        try:
            self.index = get_index()
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise NVIDIAServiceError("NVIDIA_API_KEY environment variable not set")
            
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
        except PineconeConnectionError as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA services: {str(e)}")
            raise NVIDIAServiceError(f"Service initialization failed: {str(e)}")

    async def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        try:
            results = self.index.query(
                vector=self._get_query_embedding(query),
                top_k=top_k * 2,
                include_metadata=True
            )
            
            reranked = await self._rerank_results(query, results.matches)
            return reranked[:top_k]
            
        except OpenAIError as e:
            logger.error(f"NVIDIA API error during retrieval: {str(e)}")
            raise NVIDIAServiceError(f"Retrieval failed: {str(e)}")
        except PineconeConnectionError as e:
            logger.error(f"Pinecone error during retrieval: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {str(e)}")
            raise NVIDIAServiceError(f"Retrieval failed: {str(e)}")

    def _get_query_embedding(self, query: str):
        try:
            response = self.client.embeddings.create(
                model="nvidia/llama-3.2-nv-embedqa-1b-v2",
                input=query
            )
            return response.data[0].embedding
        except OpenAIError as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            raise NVIDIAServiceError(f"Embedding generation failed: {str(e)}")

class UnstructuredRAG:
    """Handles unstructured data retrieval and processing"""
    
    def __init__(self):
        logger.info("Initializing UnstructuredRAG")
        # Add any initialization code here
        pass

    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query and return relevant results"""
        # Add query processing logic here
        return {"status": "success", "results": []} 