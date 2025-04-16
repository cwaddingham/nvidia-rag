# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the Main module for RAG ingestion pipeline integration
"""
import os
import asyncio
from typing import (
    List,
    Dict,
    Union,
    Any
)
import logging
from overrides import overrides
from datetime import datetime, timezone

from .base import BaseIngestor
from src.utils import (
    get_config,
    get_vectorstore,
    get_embedding_model,
    get_nv_ingest_client,
    get_nv_ingest_ingestor,
    get_minio_operator,
    get_unique_thumbnail_id,
    get_unique_thumbnail_id_file_name_prefix,
    get_unique_thumbnail_id_collection_prefix,
    ENABLE_NV_INGEST_VDB_UPLOAD
)

from src.utils.pinecone_utils import (
    get_pinecone_client,
    create_index,
    delete_index,
    list_indexes,
    add_documents,
    get_index_stats
)

from .document import Document

# Initialize global objects
logger = logging.getLogger(__name__)

SETTINGS = get_config()
DOCUMENT_EMBEDDER = get_embedding_model(model=SETTINGS.embeddings.model_name)
NV_INGEST_CLIENT_INSTANCE = get_nv_ingest_client()
MINIO_OPERATOR = get_minio_operator()

class NVIngestIngestor(BaseIngestor):
    """
    Main Class for RAG ingestion pipeline integration for NV-Ingest
    """

    _config = get_config()
    _vdb_upload_bulk_size = 500

    @overrides
    async def ingest_docs(
        self,
        filepaths: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main function called by ingestor server to ingest
        the documents to vector-DB

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - kwargs: Any - Metadata about the file paths
        """

        logger.info("Performing ingestion for filepaths: %s in collection_name: %s",
                    filepaths, kwargs.get("collection_name"))

        try:
            pc = get_pinecone_client()
            
            # Check if index exists
            if kwargs.get("collection_name") not in list_indexes(pc):
                raise ValueError(f"Index {kwargs.get('collection_name')} does not exist in Pinecone. Create it first using POST /collections endpoint.")

            await self._nv_ingest_ingestion(
                filepaths=filepaths,
                **kwargs
            )

            # Generate response dictionary
            uploaded_documents = [
                {
                    "document_id": f"uri://{os.path.basename(filepath)}#0",  # Base document ID, chunk IDs will be #1, #2, etc
                    "document_name": os.path.basename(filepath),
                    "size_bytes": os.path.getsize(filepath)
                }
                for filepath in filepaths
            ]

             # Get current timestamp in ISO format
            timestamp = datetime.now(timezone.utc).isoformat()
            # TODO: Store document_id, timestamp and document size as metadata

            response_data = {
                "message": "Document upload job successfully completed.",
                "total_documents": len(filepaths),
                "documents": uploaded_documents
            }

            return response_data

        except Exception as e:
            logger.error("Ingestion failed due to error: %s", e)
            from traceback import print_exc
            print_exc()
            return {"message": f"Ingestion failed due to error: {e}", "total_documents": 0, "documents": []}


    @staticmethod
    def create_collections(
        collection_names: List[str], embedding_dimension: int
    ) -> str:
        """Creates new indexes in Pinecone if they don't exist"""
        try:
            pc = get_pinecone_client()
            for name in collection_names:
                if name not in list_indexes(pc):
                    create_index(
                        pc,
                        name,
                        dimension=embedding_dimension,
                        metric=os.getenv("PINECONE_METRIC", "cosine")
                    )
            return {"message": f"Successfully created indexes: {collection_names}"}
        except Exception as e:
            raise Exception(f"Failed to create Pinecone index: {str(e)}")


    @staticmethod
    def delete_collections(
        vdb_endpoint: str, collection_names: List[str],
    ) -> Dict[str, Any]:
        """Deletes indexes from Pinecone"""
        try:
            pc = get_pinecone_client()
            
            for name in collection_names:
                if name in list_indexes(pc):
                    delete_index(pc, name)
                
            return {"message": f"Successfully deleted indexes: {collection_names}"}
        except Exception as e:
            raise Exception(f"Failed to delete Pinecone index: {str(e)}")


    @staticmethod
    def get_collections(vdb_endpoint: str) -> Dict[str, Any]:
        """
        Get index statistics from Pinecone.

        Args:
            vdb_endpoint (str): Not used for Pinecone but kept for interface compatibility

        Returns:
            Dict[str, Any]: A dictionary containing index statistics and namespace information
        """
        try:
            pc = get_pinecone_client()
            index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
            
            if index_name not in list_indexes(pc):
                return {
                    "message": f"Index {index_name} does not exist",
                    "collections": [],
                    "total_collections": 0
                }
            
            stats = get_index_stats(pc, index_name)
            
            return {
                "message": "Collections listed successfully.",
                "collections": [
                    {
                        "name": index_name,
                        "vector_count": stats["total_vector_count"],
                        "dimension": stats["dimension"],
                        "namespaces": stats["namespaces"],
                        "index_fullness": stats["index_fullness"],
                        "metric": stats["metric"]
                    }
                ],
                "total_collections": len(stats["namespaces"])
            }

        except Exception as e:
            logger.error(f"Failed to retrieve collections: {e}")
            return {
                "message": f"Failed to retrieve collections due to error: {str(e)}",
                "collections": [],
                "total_collections": 0
            }


    @staticmethod
    def get_documents(collection_name: str) -> Dict[str, Any]:
        """Lists all documents in the Pinecone index"""
        try:
            pc = get_pinecone_client()
            index = pc.Index(collection_name)
            
            # Use Pinecone's list mechanism to get all vectors
            response = index.list(
                limit=10000,  # Adjust based on expected document count
                include_metadata=True
            )
            
            # Format response
            doc_list = []
            seen_docs = set()  # Track unique document IDs
            
            for vector in response.vectors:
                doc_id = vector.metadata.get("document_id", "")
                base_doc_id = doc_id.split("#")[0]  # Get base document ID without chunk number
                
                if base_doc_id not in seen_docs:
                    seen_docs.add(base_doc_id)
                    doc_list.append({
                        "document_id": base_doc_id,
                        "document_name": vector.metadata.get("filename", ""),
                        "timestamp": vector.metadata.get("timestamp", ""),
                        "size_bytes": vector.metadata.get("size", 0)
                    })

            return {
                "documents": doc_list,
                "total_documents": len(doc_list),
                "message": "Document listing successfully completed."
            }
        except Exception as e:
            logger.exception(f"Failed to retrieve documents: {e}")
            return {
                "documents": [], 
                "total_documents": 0, 
                "message": f"Document listing failed: {e}"
            }


    @staticmethod
    def delete_documents(document_names: List[str], document_ids: List[str], collection_name: str, vdb_endpoint: str) -> Dict[str, Any]:
        """Delete documents from Pinecone and Minio"""
        try:
            if not document_names and not document_ids:
                raise ValueError("No document names or IDs provided for deletion")

            pc = get_pinecone_client()
            index = pc.Index(collection_name)

            # Build filter for deletion
            filter_conditions = []
            if document_names:
                filter_conditions.append({"filename": {"$in": document_names}})
            if document_ids:
                filter_conditions.append({"document_id": {"$in": document_ids}})
            
            # Combine conditions with OR
            delete_filter = {"$or": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
            
            # Delete from Pinecone
            index.delete(filter=delete_filter)

            # Delete from Minio
            for doc in document_names:
                filename_prefix = get_unique_thumbnail_id_file_name_prefix(collection_name, doc)
                delete_object_names = MINIO_OPERATOR.list_payloads(filename_prefix)
                MINIO_OPERATOR.delete_payloads(delete_object_names)

            return {
                "message": "Files deleted successfully",
                "total_documents": len(document_names) + len(document_ids),
                "documents": [{"document_name": doc} for doc in document_names]
            }

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return {
                "message": f"Failed to delete files: {e}",
                "total_documents": 0,
                "documents": []
            }

    @staticmethod
    def _prepare_metadata(
        result_element: Dict[str, Union[str, dict]]
    ) -> Dict[str, str]:
        """
        Only used if ENABLE_NV_INGEST_VDB_UPLOAD=False
        Prepare metadata object w.r.t. to a single chunk

        Arguments:
            - result_element: Dict[str, Union[str, dict]]] - Result element for single chunk

        Returns:
            - metadata: Dict[str, str] - Dict of metadata for s single chunk
            {
                "source": "<filepath>",
                "chunk_type": "<chunk_type>", # ["text", "image", "table", "chart"]
                "source_name": "<filename>",
                "content": "<base64_str encoded content>" # Only for ["image", "table", "chart"]
            }
        """
        source_id = result_element.get("metadata").get("source_metadata").get("source_id")

        # Get chunk_type
        if result_element.get("document_type") == "structured":
            chunk_type = result_element.get("metadata").get("content_metadata").get("subtype")
        else:
            chunk_type = result_element.get("document_type")

        # Get base64_str encoded content, empty str in case of text
        content = result_element.get("metadata").get("content") if chunk_type != "text" else ""

        metadata = {
            "source": source_id, # Add filepath (Key-name same for backward compatibility)
            "chunk_type": chunk_type, # ["text", "image", "table", "chart"]
            "source_name": os.path.basename(source_id), # Add filename
            # "content": content # content encoded in base64_str format [Must not exceed 64KB]
        }
        return metadata

    def _prepare_documents(self, results: List[List[Dict[str, Union[str, dict]]]]) -> List[Document]:
        """Prepare documents from nv-ingest results"""
        documents = []
        for result in results:
            for idx, result_element in enumerate(result, 1):  # Start chunk numbering at 1
                if result_element.get("document_type") == "text":
                    source_id = result_element.get("metadata", {}).get("source_metadata", {}).get("source_id", "")
                    base_name = os.path.basename(source_id)
                    
                    # Create semantic document ID with chunk number
                    doc_id = f"uri://{base_name}#{idx}"
                    
                    # Get page number if available
                    page_number = result_element.get("metadata", {}).get("content_metadata", {}).get("page_number", 0)
                    
                    documents.append(
                        Document(
                            content=result_element.get("content"),
                            metadata={
                                "document_id": doc_id,
                                "filename": base_name,
                                "chunk": idx,
                                "page": page_number,
                                "source": source_id
                            }
                        )
                    )
        return documents

    def _add_documents_to_vectorstore(
        self,
        documents: List[Document],
        collection_name: str
    ) -> None:
        """
        Add documents to Pinecone index
        
        Arguments:
            - documents: List[Document] - List of documents to add
            - collection_name: str - Pinecone index name
            - vdb_endpoint: str - Not used for Pinecone but kept for interface compatibility
        """
        pc = get_pinecone_client()
        add_documents(
            pc=pc,
            index_name=collection_name,
            documents=documents,
            embedder=DOCUMENT_EMBEDDER,
            batch_size=self._vdb_upload_bulk_size
        )

    @staticmethod
    def _put_content_to_minio(
        results: List[List[Dict[str, Union[str, dict]]]],
        collection_name: str,
    ) -> None:
        """
        Put nv-ingest image/table/chart content to minio
        """
        if not os.getenv("ENABLE_CITATIONS", "True") in ["True", "true"]:
            logger.info(f"Skipping minio insertion for collection: {collection_name}")
            return # Don't perform minio insertion if captioning is disabled

        for result in results:
            for result_element in result:
                if result_element.get("document_type") in ["image", "structured"]:
                    # Pull content from result_element
                    content = result_element.get("metadata").get("content")
                    file_name = os.path.basename(result_element.get("metadata").get("source_metadata").get("source_id"))
                    page_number = result_element.get("metadata").get("content_metadata").get("page_number")
                    location = result_element.get("metadata").get("content_metadata").get("location")

                    if location is not None:
                        # Get unique_thumbnail_id
                        unique_thumbnail_id = get_unique_thumbnail_id(
                            collection_name=collection_name,
                            file_name=file_name,
                            page_number=page_number,
                            location=location
                        )
                        # Put payload to minio
                        MINIO_OPERATOR.put_payload(
                            payload={"content": content},
                            object_name=unique_thumbnail_id
                        )

    async def _nv_ingest_ingestion(
        self,
        filepaths: List[str],
        **kwargs
    ) -> None:
        """
        This methods performs following steps:
        - Perform extraction and splitting using NV-ingest ingestor
        - Prepare langchain documents from the nv-ingest results
        - Embeds and add documents to Vectorstore collection

        Arguments:
            - filepaths: List[str] - List of absolute filepaths
            - kwargs: Any - Metadata about the file paths
        """
        nv_ingest_ingestor = get_nv_ingest_ingestor(
            nv_ingest_client_instance=NV_INGEST_CLIENT_INSTANCE,
            filepaths=filepaths,
            **kwargs
        )
        logger.info(f"Performing ingestion with parameters: {kwargs}")
        results = await asyncio.to_thread(nv_ingest_ingestor.ingest)
        logger.debug("NV-ingest Job for collection_name: %s is complete!", kwargs.get("collection_name"))

        if not results:
            error_message = "NV-Ingest ingestion failed with no results. Please check the ingestor-server microservice logs for more details."
            logger.error(error_message)
            raise Exception(error_message)

        self._put_content_to_minio(
            results=results,
            collection_name=kwargs.get("collection_name")
        )

        if not ENABLE_NV_INGEST_VDB_UPLOAD:
            logger.debug("Performing embedding and vector DB upload")

            # Prepare the documents for nv-ingest results
            documents = self._prepare_documents(results)

            # Add all documents to VectorStore
            self._add_documents_to_vectorstore(
                documents=documents,
                collection_name=kwargs.get("collection_name"),
                vdb_endpoint=kwargs.get("vdb_endpoint")
            )
            logger.debug("Vector DB upload complete to: %s in collection %s", kwargs.get("vdb_endpoint"), kwargs.get("collection_name"))