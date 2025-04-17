from typing import List
import os
import logging
import time
from pathlib import Path
from http import HTTPStatus
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.rest.rest_client import RestClient
from requests.exceptions import RequestException
from requests.exceptions import Timeout

logger = logging.getLogger(__name__)

class DocumentProcessorError(Exception):
    """Base exception for document processing errors"""
    pass

class AuthenticationError(DocumentProcessorError):
    """Raised when there are NVIDIA API authentication issues"""
    pass

class TimeoutError(DocumentProcessorError):
    """Raised when document processing times out"""
    pass

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with NV-Ingest"""
        try:
            if not os.getenv("NVIDIA_API_KEY"):
                raise AuthenticationError("NVIDIA_API_KEY environment variable not set")

            self.client = NvIngestClient(
                message_client_hostname=os.getenv("APP_NVINGEST_MESSAGECLIENTHOSTNAME", "nv-ingest-ms-runtime"),
                message_client_port=int(os.getenv("APP_NVINGEST_MESSAGECLIENTPORT", "7670"))
            )

        except RequestException as e:
            if e.response and e.response.status_code == HTTPStatus.UNAUTHORIZED:
                raise AuthenticationError("Invalid NVIDIA API key") from e
            elif e.response and e.response.status_code == HTTPStatus.FORBIDDEN:
                raise AuthenticationError("API key lacks required permissions") from e
            else:
                logger.error(f"Connection error: {str(e)}")
                raise DocumentProcessorError(f"Failed to connect to NVIDIA services: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to initialize NV-Ingest client: {str(e)}")
            raise DocumentProcessorError(f"Initialization error: {str(e)}") from e

    def process_documents(self, data_dir: str) -> List[dict]:
        """Process documents using NVIDIA's NV-Ingest pipeline"""
        try:
            if not Path(data_dir).exists():
                raise DocumentProcessorError(f"Directory not found: {data_dir}")

            logger.info(f"Starting document processing from {data_dir}")
            logger.info(f"Found files: {os.listdir(data_dir)}")

            # Set timeout for processing
            timeout = time.time() + 300  # 5 minute timeout

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
                .embed(
                    model_name="nvidia/llama-3.2-nv-embedqa-1b-v2"
                )
            )
            
            logger.info("Starting ingestion...")
            try:
                results = ingestor.ingest(show_progress=True)
                if not results:
                    raise DocumentProcessorError("No results returned from ingestion")
            except Timeout:
                raise TimeoutError("Document processing timed out after 5 minutes")
            except Exception as e:
                logger.error(f"Ingestion error: {str(e)}")
                raise DocumentProcessorError(f"Ingestion failed: {str(e)}")

            if time.time() > timeout:
                raise TimeoutError("Document processing timed out after 5 minutes")

            logger.info(f"Ingestion complete. Results: {results}")
            return results
        except AuthenticationError:
            raise  # Re-raise auth errors
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise DocumentProcessorError(f"Document processing error: {str(e)}") from e 