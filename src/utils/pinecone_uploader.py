from typing import List, Dict
from .pinecone_utils import get_index

class PineconeUploader:
    def __init__(self):
        self.index = get_index()

    def upload_vectors(self, documents: List[Dict], namespace: str = "default"):
        """Upload document vectors to Pinecone"""
        vectors = []
        
        for doc in documents:
            vectors.append({
                'id': doc['id'],
                'values': doc['embedding'],
                'metadata': {
                    'text': doc['text'],
                    'filename': doc['filename'],
                    'page': doc.get('page', 0)
                }
            })
            
        self.index.upsert(vectors=vectors, namespace=namespace) 