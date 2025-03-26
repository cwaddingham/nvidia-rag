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
"""Base interface that all RAG examples should implement."""

from abc import ABC
from abc import abstractmethod
from typing import Generator, Dict, Any, Optional
from typing import List

from pinecone import Pinecone, ServerlessSpec
from .server import Message


class BaseExample(ABC):
    """This class defines the basic structure for building RAG server examples."""

    def __init__(self):
        """Initialize the BaseExample."""
        self.pc = None
        self.index = None

    def init_pinecone(self, api_key: str, environment: Optional[str] = None) -> None:
        """Initialize Pinecone client and connect to index.
        
        Args:
            api_key (str): Pinecone API key
            environment (Optional[str]): Pinecone environment (not needed for serverless)
        """
        self.pc = Pinecone(api_key=api_key)

    def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        serverless: bool = False,
        cloud: str = "aws",
        region: str = "us-west-2"
    ) -> None:
        """Create a Pinecone index if it doesn't exist.
        
        Args:
            index_name (str): Name of the index
            dimension (int): Dimension of vectors
            metric (str): Distance metric (cosine, euclidean, dotproduct)
            serverless (bool): Whether to use serverless
            cloud (str): Cloud provider for serverless
            region (str): Region for serverless
        """
        if not self.pc:
            raise RuntimeError("Pinecone client not initialized")
            
        # Check if index exists
        if index_name not in [index.name for index in self.pc.list_indexes()]:
            if serverless:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
            else:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric
                )
                
        self.index = self.pc.Index(index_name)

    @abstractmethod
    def llm_chain(self, query: str, chat_history: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Implements the LLM chain logic specific to the example.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `False`.

        Args:
            query (str): Query to be answered by llm.
            chat_history (List[Message]): Conversation history between user and chain.
            kwargs: kwargs

        Returns:
            Generator[str, None, None]: A generator that yields strings, representing the tokens of the LLM chain.
        """

        pass

    @abstractmethod
    def rag_chain(self, query: str, chat_history: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Implements the RAG chain logic specific to the example.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True`.

        Args:
            query (str): Query to be answered by llm.
            chat_history (List[Message]): Conversation history between user and chain.
            kwargs: kwargs

        Returns:
            Generator[str, None, None]: Represents the steps or outputs of the RAG chain.
        """

        pass

    @abstractmethod
    def ingest_docs(self, data_dir: str, filename: str, collection_name: str) -> None:
        """Defines how documents are ingested for processing.
        
        Args:
            data_dir (str): Directory containing the document
            filename (str): Name of the document file
            collection_name (str): Name of the Pinecone index to use
        """
        pass

    def delete_documents(self, filenames: List[str], collection_name: str) -> bool:
        """Delete documents from the vector store.
        
        Args:
            filenames (List[str]): List of filenames to delete
            collection_name (str): Name of the Pinecone index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.index:
                self.index = self.pc.Index(collection_name)
                
            # Delete vectors by metadata filter
            self.index.delete(
                filter={"filename": {"$in": filenames}}
            )
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def get_documents(self, collection_name: str) -> List[str]:
        """Get list of documents in the vector store.
        
        Args:
            collection_name (str): Name of the Pinecone index
            
        Returns:
            List[str]: List of filenames
        """
        try:
            if not self.index:
                self.index = self.pc.Index(collection_name)
                
            # Query with empty vector to get metadata
            results = self.index.query(
                vector=[0] * self.index.describe_index_stats()['dimension'],
                top_k=10000,
                include_metadata=True
            )
            
            # Extract unique filenames from metadata
            filenames = set()
            for match in results.matches:
                if match.metadata and 'filename' in match.metadata:
                    filenames.add(match.metadata['filename'])
                    
            return list(filenames)
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
