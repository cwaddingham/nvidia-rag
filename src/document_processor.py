from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"):
        self.client = NvIngestClient(
            message_client_allocator=SimpleClient,
            message_client_port=7671,
            message_client_hostname="localhost"
        )
        self.embedding_model = embedding_model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_documents(self, data_dir: str, collection_name: str) -> List[dict]:
        """Process documents using NVIDIA's nv-ingest pipeline"""
        ingestor = (
            Ingestor(client=self.client)
            .files(data_dir)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                text_depth="page"
            )
            .embed()
        )
        
        results = ingestor.ingest(show_progress=True)
        return results 