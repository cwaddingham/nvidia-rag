from typing import List
from pinecone import Pinecone

class PineconeEmbedder:
    """Embedder that uses Pinecone's hosted inference service"""
    
    def __init__(self, model_name: str, input_type: str = "passage", truncate: str = "END"):
        self.pc = Pinecone()  # Uses API key from environment
        self.model_name = model_name
        self.input_type = input_type
        self.truncate = truncate
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        return self.pc.inference.embed(
            model=self.model_name,
            inputs=texts,
            parameters={
                "input_type": self.input_type,
                "truncate": self.truncate
            }
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.pc.inference.embed(
            model=self.model_name,
            inputs=[text],
            parameters={
                "input_type": "query",  # Always use query type for queries
                "truncate": self.truncate
            }
        )[0]  # Return first (and only) embedding 