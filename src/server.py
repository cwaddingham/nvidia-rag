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
"""The definition of the NVIDIA RAG server which acts as the main orchestrator."""
import asyncio
import logging
import os
import time
from inspect import getmembers
from inspect import isclass
from typing import Any, Literal, Optional
from typing import Dict
from typing import List
from typing import Union
from uuid import uuid4
import signal
import sys

import bleach
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import Field
from pydantic import constr
from pydantic import validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from fastapi.staticfiles import StaticFiles

from pinecone import Pinecone, ServerlessSpec, PineconeException
from document_processor import DocumentProcessor
from utils.pinecone_utils import get_pinecone_client, get_index
from tracing import instrument
from configuration import get_config

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)

settings = get_config()
default_max_tokens = 128000 if "deepseek-r1" in str(settings.llm.model_name) else 1024
default_temperature = 0.6 if "deepseek-r1" in str(settings.llm.model_name) else 0.2
default_top_p = 0.95 if "deepseek-r1" in str(settings.llm.model_name) else 0.7

tags_metadata = [
    {
        "name": "Health APIs",
        "description": "APIs for checking and monitoring server liveliness and readiness.",
    },
    {"name": "Retrieval APIs", "description": "APIs for retrieving document chunks for a query."},
    {"name": "RAG APIs", "description": "APIs for retrieval followed by generation."},
]

# create the FastAPI server
app = FastAPI(
    title="NVIDIA RAG Server",
    openapi_tags=tags_metadata
)

# Allow access in browser from RAG UI and Storybook (development)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

EXAMPLE_DIR = "./"
MINIO_OPERATOR = get_minio_operator()
FALLBACK_EXCEPTION_MSG = "Error from rag-server. Please check rag-server logs for more details."

# Log server initialization details first
logger.info("Initializing NVIDIA RAG server...")

UNSTRUCTURED_RAG = UnstructuredRAG()

settings = get_config()
metrics = None
if settings.tracing.enabled:
    metrics = instrument(app, settings)

class Message(BaseModel):
    """Definition of the Chat Message type."""

    role: Literal["user", "assistant", "system", None] = Field(
        description="Role for a message: either 'user' or 'assistant' or 'system",
        default="user"
    )
    content: str = Field(
        description="The input query/prompt to the pipeline.",
        default="Hello! What can you help me with?",
        max_length=131072,
        pattern=r'[\s\S]*',
    )

    @validator('role')
    @classmethod
    def validate_role(cls, value):
        """ Field validator function to validate values of the field role"""
        if value:
            value = bleach.clean(value, strip=True)
            valid_roles = {'user', 'assistant', 'system'}
            if value is not None and value.lower() not in valid_roles:
                raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
            return value.lower()

    @validator('content')
    @classmethod
    def sanitize_content(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        return bleach.clean(v, strip=True)


class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    messages: List[Message] = Field(
        ...,
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    use_knowledge_base: bool = Field(default=True, description="Whether to use a knowledge base")
    temperature: float = Field(
        default_temperature,
        description="The sampling temperature to use for text generation. "
        "The higher the temperature value is, the less deterministic the output text will be. "
        "It is not recommended to modify both temperature and top_p in the same call.",
        ge=0.1,
        le=1.0,
    )
    top_p: float = Field(
        default_top_p,
        description="The top-p sampling mass used for text generation. "
        "The top-p value determines the probability mass that is sampled at sampling time. "
        "For example, if top_p = 0.2, only the most likely tokens "
        "(summing to 0.2 cumulative probability) will be sampled. "
        "It is not recommended to modify both temperature and top_p in the same call.",
        ge=0.1,
        le=1.0,
    )
    max_tokens: int = Field(
        default_max_tokens,
        description="The maximum number of tokens to generate in any given call. "
        "Note that the model is not aware of this value, "
        " and generation will simply stop at the number of tokens specified.",
        ge=0,
        le=128000,
        format="int64",
    )
    reranker_top_k: int = Field(
        description="The maximum number of documents to return in the response.",
        default=int(os.getenv("APP_RETRIEVER_TOPK", 4)),
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=int(os.getenv("VECTOR_DB_TOPK", 4)),
        ge=0,
        le=400,
        format="int64",
    )
    # Reserved for future use
    # vdb_search_type: str = Field(
    #     description="Search type for the vector space. Can be one of dense or hybrid",
    #     default=os.getenv("APP_VECTORSTORE_SEARCHTYPE", "dense")
    # )
    vdb_endpoint: str = Field(
        description="Endpoint url of the vector database server.",
        default=os.getenv("APP_VECTORSTORE_URL", "http://localhost:19530")
    )
    collection_name: str = Field(
        description="Name of collection to be used for inference.",
        default=os.getenv("COLLECTION_NAME", ""),
        max_length=4096,
        pattern=r'[\s\S]*',
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=os.getenv("ENABLE_QUERYREWRITER", "False").lower() in ["true", "True"],
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=os.getenv("ENABLE_RERANKER", "True").lower() in ["true", "True"],
    )
    enable_guardrails: bool = Field(
        description="Enable or disable guardrailing of queries/responses.",
        default=os.getenv("ENABLE_GUARDRAILS", "False").lower() in ["true", "True"],
    )
    enable_citations: bool = Field(
        description="Enable or disable citations as part of response.",
        default=os.getenv("ENABLE_CITATIONS", "True").lower() in ["true", "True"],
    )
    model: str = Field(
        description="Name of NIM LLM model to be used for inference.",
        default=os.getenv("APP_LLM_MODELNAME", "").strip('"'),
        max_length=4096,
        pattern=r'[\s\S]*',
    )
    llm_endpoint: str = Field(
        description="Endpoint URL for the llm model server.",
        default=os.getenv("APP_LLM_SERVERURL", "").strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=os.getenv("APP_EMBEDDINGS_MODELNAME", "").strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: Optional[str] = Field(
        description="Endpoint URL for the embedding model server.",
        default=os.getenv("APP_EMBEDDINGS_SERVERURL", "").strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=os.getenv("APP_RANKING_MODELNAME", "").strip('"'),
        max_length=256,
    )
    reranker_endpoint: Optional[str] = Field(
        description="Endpoint URL for the reranker model server.",
        default=os.getenv("APP_RANKING_SERVERURL", "").strip('"'),
        max_length=2048,
    )

    # seed: int = Field(42, description="If specified, our system will make a best effort to sample deterministically,
    #       such that repeated requests with the same seed and parameters should return the same result.")
    # bad: List[str] = Field(None, description="A word or list of words not to use. The words are case sensitive.")
    stop: List[constr(max_length=256)] = Field(
        description="A string or a list of strings where the API will stop generating further tokens."
        "The returned text will not contain the stop sequence.",
        max_items=256,
        default=[],
    )
    # stream: bool = Field(True, description="If set, partial message deltas will be sent.
    #           Tokens will be sent as data-only server-sent events (SSE) as they become available
    #           (JSON responses are prefixed by data:), with the stream terminated by a data: [DONE] message.")

    @validator('use_knowledge_base')
    @classmethod
    def sanitize_use_kb(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        v = bleach.clean(str(v), strip=True)
        try:
            return {"True": True, "False": False}[v]
        except KeyError as e:
            raise ValueError("use_knowledge_base must be a boolean value") from e

    @validator('temperature')
    @classmethod
    def sanitize_temperature(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        return float(bleach.clean(str(v), strip=True))

    @validator('top_p')
    @classmethod
    def sanitize_top_p(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        return float(bleach.clean(str(v), strip=True))

    # Validator to normalize model information
    @field_validator("reranker_endpoint", "embedding_endpoint", "embedding_model", "reranker_model", "model", "llm_endpoint", mode="before")
    @classmethod
    def normalize_model_info(cls, value):
        if isinstance(value, str):
            return value.strip(" ").strip('"')
        return value

    @field_validator("reranker_top_k")
    @classmethod
    def validate_reranker_top_k(cls, reranker_top_k, info):
        vdb_top_k = info.data.get("vdb_top_k")  # Use `info.data`
        if vdb_top_k is not None and reranker_top_k > vdb_top_k:
            raise ValueError("reranker_top_k must be less than or equal to vdb_top_k")
        return reranker_top_k


class ChainResponseChoices(BaseModel):
    """ Definition of Chain response choices"""

    index: int = Field(default=0, ge=0, le=256, format="int64")
    message: Message = Field(default=Message(role="assistant", content=""))
    delta: Message = Field(default=Message(role=None, content=""))
    finish_reason: Optional[str] = Field(default=None, max_length=4096, pattern=r'[\s\S]*')


class Usage(BaseModel):
    """Token usage information."""

    total_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Total tokens used in the request",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Tokens used for the prompt",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Tokens used for the completion",
    )

class SourceMetadata(BaseModel):
    """Metadata associated with a document source."""

    language: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Language of the document",
    )
    date_created: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Creation date of the document",
    )
    last_modified: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Last modification date",
    )
    page_number: int = Field(
        default=0,
        ge=-1,
        le=1000000,
        format="int64",
        description="Page number in the document",
    )
    description: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Description of the document content",
    )
    height: int = Field(
        default=0,
        ge=0,
        le=100000,
        format="int64",
        description="Height of the document in pixels",
    )
    width: int = Field(
        default=0,
        ge=0,
        le=100000,
        format="int64",
        description="Width of the document in pixels",
    )
    location: List[float] = Field(
        default=[],
        description="Bounding box location of the content"
    )
    location_max_dimensions: List[int] = Field(
        default=[],
        description="Maximum dimensions of the document"
    )

class SourceResult(BaseModel):
    """Represents a single source document result."""

    document_id: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Unique identifier of the document",
    )
    content: str = Field(
        default="",
        pattern=r"[\s\S]*",
        description="Extracted content from the document",
    )
    document_name: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Name of the document",
    )
    document_type: Literal["image", "text", "table", "chart"] = Field(
        default="text",
        description="Type of document content"
    )
    score: float = Field(
        default=0.0,
        description="Relevance score of the document")

    metadata: SourceMetadata

class Citations(BaseModel):
    """Represents the sources section of the API response."""

    total_results: int = Field(
        default=0,
        ge=0,
        le=1000000,
        format="int64",
        description="Total number of source documents found",
    )
    results: List[SourceResult] = Field(
        default=[], description="List of document results"
    )

class ChainResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""

    id: str = Field(default="", max_length=100000, pattern=r'[\s\S]*')
    choices: List[ChainResponseChoices] = Field(default=[], max_items=256)
    # context will be deprecated once `sources` field is implemented and populated
    model: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')
    object: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')
    created: int = Field(default=0, ge=0, le=9999999999, format="int64")
    # Place holder fields for now to match generate API response structure
    usage: Optional[Usage] = Field(default=Usage(), description="Token usage statistics")
    citations: Optional[Citations] = Field(default=Citations(), description="Source documents used for the response")


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    query: str = Field(
        description="The content or keywords to search for within documents.",
        max_length=131072,
        pattern=r'[\s\S]*',
        default="",
    )
    reranker_top_k: int = Field(
        description="Number of document chunks to retrieve.",
        default=int(os.getenv("APP_RETRIEVER_TOPK", 4)),
        ge=0,
        le=25,
        format="int64",
    )
    vdb_top_k: int = Field(
        description="Number of top results to retrieve from the vector database.",
        default=int(os.getenv("VECTOR_DB_TOPK", 4)),
        ge=0,
        le=400,
        format="int64",
    )
    vdb_endpoint: str = Field(
        description="Endpoint url of the vector database server.",
        default=os.getenv("APP_VECTORSTORE_URL", "http://localhost:19530")
    )
    # Reserved for future use
    # vdb_search_type: str = Field(
    #     description="Search type for the vector space. Can be one of dense or hybrid",
    #     default=os.getenv("APP_VECTORSTORE_SEARCHTYPE", "dense")
    # )
    collection_name: str = Field(
        description="Name of collection to be used for searching document.",
        default=os.getenv("COLLECTION_NAME", ""),
        max_length=4096,
        pattern=r'[\s\S]*',
    )
    messages: List[Message] = Field(
        ...,
        description="A list of messages comprising the conversation so far. "
        "The roles of the messages must be alternating between user and assistant. "
        "The last input message should have role user. "
        "A message with the the system role is optional, and must be the very first message if it is present.",
        max_items=50000,
    )
    enable_query_rewriting: bool = Field(
        description="Enable or disable query rewriting.",
        default=os.getenv("ENABLE_QUERYREWRITER", "True").lower() in ["true", "True"],
    )
    enable_reranker: bool = Field(
        description="Enable or disable reranking by the ranker model.",
        default=os.getenv("ENABLE_RERANKER", "True").lower() in ["true", "True"],
    )
    embedding_model: str = Field(
        description="Name of the embedding model used for vectorization.",
        default=os.getenv("APP_EMBEDDINGS_MODELNAME", "").strip('"'),
        max_length=256,  # Reduced from 4096 as model names are typically short
    )
    embedding_endpoint: str = Field(
        description="Endpoint URL for the embedding model server.",
        default=os.getenv("APP_EMBEDDINGS_SERVERURL", "").strip('"'),
        max_length=2048,  # URLs can be long, but 4096 is excessive
    )
    reranker_model: str = Field(
        description="Name of the reranker model used for ranking results.",
        default=os.getenv("APP_RANKING_MODELNAME", "").strip('"'),
        max_length=256,
    )
    reranker_endpoint: Optional[str] = Field(
        description="Endpoint URL for the reranker model server.",
        default=os.getenv("APP_RANKING_SERVERURL", "").strip('"'),
        max_length=2048,
    )

    # Validator to normalize model information
    @field_validator("reranker_endpoint", "embedding_endpoint", "embedding_model", "reranker_model", mode="before")
    @classmethod
    def normalize_model_info(cls, value):
        if isinstance(value, str):
            return value.strip().strip('"')
        return value


# Define the service health models in server.py
class BaseServiceHealthInfo(BaseModel):
    """Base health info model with common fields for all services"""
    service: str
    url: str
    status: str
    latency_ms: float = 0
    error: Optional[str] = None

class DatabaseHealthInfo(BaseServiceHealthInfo):
    """Health info specific to database services"""
    collections: Optional[int] = None

class StorageHealthInfo(BaseServiceHealthInfo):
    """Health info specific to object storage services"""
    buckets: Optional[int] = None
    message: Optional[str] = None

class NIMServiceHealthInfo(BaseServiceHealthInfo):
    """Health info specific to NIM services (LLM, embeddings, etc.)"""
    message: Optional[str] = None
    http_status: Optional[int] = None

class HealthResponse(BaseModel):
    message: str = Field(max_length=4096, pattern=r'[\s\S]*', default="")


@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone on startup"""
    pc = get_pinecone_client()
    app.state.pinecone = pc

    # Create index if it doesn't exist
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
    metric = os.getenv("PINECONE_METRIC", "cosine")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-west-2")
    
    # Check if index exists using has_index
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            vector_type="dense",
            deletion_protection="disabled",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    
    # Get index for use in app
    app.state.index = pc.Index(index_name)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": jsonable_encoder(exc.errors(), exclude={"input"})},
    )


def error_response_generator(exception_msg: str):
    """
    Generate a stream of data for the error response
    """

    def get_chain_response(
        content: str = "",
        finish_reason: Union[str, None] = None
    ) -> ChainResponse:
        """
        Get a chain response for an exception
        Args:
            exception_msg: str - Exception message
        Returns:
            chain_response: ChainResponse - Chain response for an exception
        """
        chain_response = ChainResponse()
        chain_response.id = str(uuid4())
        response_choice = ChainResponseChoices(
            index=0,
            message=Message(role="assistant", content=content),
            delta=Message(role=None, content=content),
            finish_reason=finish_reason
        )
        chain_response.choices.append(response_choice)  # pylint: disable=E1101
        chain_response.object = "chat.completion.chunk"
        chain_response.created = int(time.time())
        return chain_response

    for i in range(0, len(exception_msg), 5):
        exception_msg_content = exception_msg[i:i+5]
        chain_response = get_chain_response(content=exception_msg_content)
        yield "data: " + str(chain_response.model_dump_json()) + "\n\n"
    chain_response = get_chain_response(finish_reason="stop")
    yield "data: " + str(chain_response.model_dump_json()) + "\n\n"


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health APIs"],
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def health_check():
    """Health check endpoint."""
    try:
        # Add basic service checks here
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/documents", tags=["Ingestion APIs"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for ingestion"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        processor = DocumentProcessor()
        # Save files temporarily
        temp_dir = Path("tmp-data/uploaded_files")
        # Clean directory if it exists
        if temp_dir.exists():
            logger.info(f"Cleaning existing files from {temp_dir}")
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Uploading files to {temp_dir}")
        for file in files:
            file_path = temp_dir / file.filename
            logger.info(f"Writing file: {file.filename}")
            with open(file_path, "wb") as f:
                f.write(await file.read())
                
        # Process documents
        try:
            results = processor.process_documents(str(temp_dir))
            if not results:
                raise HTTPException(status_code=500, detail="Document processing completed but no results were returned")
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except DocumentProcessorError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        # Clean up
        shutil.rmtree(temp_dir)
        return {"message": "Documents processed successfully"}
    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate",
    tags=["RAG APIs"],
    response_model=ChainResponse,
    responses={
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The client cancelled the request"
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def generate_answer(request: Request, prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        chat_history = prompt.messages
        collection_name = prompt.collection_name
        
        # Helper function to escape JSON-like structures in content
        def escape_json_content(content: str) -> str:
            """Escape curly braces in content to avoid JSON parsing issues"""
            return content.replace("{", "{{").replace("}", "}}")
        
        # The last user message will be the query for the rag or llm chain
        last_user_message = next((message.content for message in reversed(chat_history) if message.role == 'user'),
                                None)
        if last_user_message:
            last_user_message = escape_json_content(last_user_message)

        # Process chat history and escape JSON-like structures
        processed_chat_history = []
        for message in chat_history:
            if message.role == 'user':
                # Skip the last user message as it's handled separately
                continue
            # Create new Message with escaped content
            processed_message = Message(
                role=message.role,
                content=escape_json_content(message.content)
            )
            processed_chat_history.append(processed_message)

        # All the other information from the prompt like the temperature, top_p etc., are llm_settings
        kwargs = {
            key: value
            for key, value in vars(prompt).items() if key not in ['messages', 'use_knowledge_base', 'collection_name', 'vdb_top_k', 'reranker_top_k']
        }
        # pylint: disable=unreachable
        generator = None
        # call rag_chain if use_knowledge_base is enabled
        # Initialize contexts variable before branching
        contexts = list()

        if prompt.use_knowledge_base:
            logger.info("Knowledge base is enabled. Using rag chain for response generation.")
            generator, contexts = UNSTRUCTURED_RAG.rag_chain(query=last_user_message,
                                          chat_history=processed_chat_history,
                                          reranker_top_k=prompt.reranker_top_k,
                                          vdb_top_k=prompt.vdb_top_k,
                                          collection_name=collection_name,
                                          **kwargs)
        else:
            generator = UNSTRUCTURED_RAG.llm_chain(query=last_user_message, chat_history=processed_chat_history, **kwargs)

        def response_generator():
            """Convert generator streaming response into `data: ChainResponse` format for chunk"""
            try:
                # unique response id for every query
                resp_id = str(uuid4())
                if generator:
                    logger.debug("Generated response chunks\n")
                    # Create ChainResponse object for every token generated
                    first_chunk = True
                    for chunk in generator:
                        # TODO: This is a hack to clear contexts if we get an error response from nemoguardrails
                        if chunk == "I'm sorry, I can't respond to that.":
                            # Clear contexts if we get an error response
                            nonlocal contexts
                            contexts = list()
                        chain_response = ChainResponse()
                        response_choice = ChainResponseChoices(
                            index=0,
                            message=Message(role="assistant", content=chunk),
                            delta=Message(role=None, content=chunk),
                            finish_reason=None

                        )
                        chain_response.id = resp_id
                        chain_response.choices.append(response_choice)  # pylint: disable=E1101
                        chain_response.model = prompt.model
                        chain_response.object = "chat.completion.chunk"
                        chain_response.created = int(time.time())
                        if first_chunk:
                            chain_response.citations = prepare_citations(
                                retrieved_documents=contexts,
                                collection_name=collection_name,
                                enable_citations=prompt.enable_citations,
                            )
                            first_chunk = False
                        logger.debug(response_choice)
                        # Send generator with tokens in ChainResponse format
                        yield "data: " + str(chain_response.json()) + "\n\n"
                    chain_response = ChainResponse()

                    # [DONE] indicate end of response from server
                    response_choice = ChainResponseChoices(
                        finish_reason="stop",
                    )
                    chain_response.id = resp_id
                    chain_response.choices.append(response_choice)  # pylint: disable=E1101
                    chain_response.model = prompt.model
                    chain_response.object = "chat.completion.chunk"
                    chain_response.created = int(time.time())
                    logger.debug(response_choice)
                    yield "data: " + str(chain_response.json()) + "\n\n"
                else:
                    chain_response = ChainResponse()
                    yield "data: " + str(chain_response.json()) + "\n\n"
            
            except Exception as e:
                logger.exception("Error from response generator in /generate endpoint. Error details: %s", e)
                yield from error_response_generator(FALLBACK_EXCEPTION_MSG)
        
        return StreamingResponse(response_generator(), media_type="text/event-stream")
        # pylint: enable=unreachable
    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during response generation. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)

    except PineconeException as e:
        exception_msg = ("Error from Pinecone server. Please ensure you have ingested some documents. "
                         "Please check rag-server logs for more details.")
        chain_response = ChainResponse()
        response_choice = ChainResponseChoices(index=0,
                                               message=Message(role="assistant", content=exception_msg),
                                               delta=Message(role="assistant", content=exception_msg),
                                               finish_reason="stop")
        chain_response.choices.append(response_choice)
        chain_response.model = prompt.model
        chain_response.object = "chat.completion.chunk"
        chain_response.created = int(time.time())
        logger.error(
            "Error from Pinecone database in /generate endpoint. Please ensure you have ingested some documents. " +
            "Error details: %s",
            e)
        return StreamingResponse(error_response_generator(exception_msg),
                                 media_type="text/event-stream",
                                 status_code=500)

# Alias function to /generate endpoint OpenAI API compatibility
@app.post(
    "/chat/completions",
    tags=["RAG APIs"],
    response_model=ChainResponse,
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def v1_chat_completions(request: Request, prompt: Prompt) -> StreamingResponse:
    """ Just an alias function to /generate endpoint which is openai compatible """

    response = await generate_answer(request, prompt)
    return response


@app.post(
    "/search",
    tags=["Retrieval APIs"],
    response_model=Citations,
    responses={
        499: {
            "description": "Client Closed Request",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The client cancelled the request"
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error occurred"
                    }
                }
            },
        }
    },
)
async def document_search(request: Request, data: DocumentSearch) -> Dict[str, List[Dict[str, Any]]]:
    """Search for the most relevant documents for the given search parameters."""

    if metrics:
        metrics.update_api_requests(method=request.method, endpoint=request.url.path)
    try:
        if hasattr(UNSTRUCTURED_RAG, "document_search") and callable(UNSTRUCTURED_RAG.document_search):

            # All the other information from the data are in kwargs like embedding model, embedding url
            excluded_keys = {"query", "reranker_top_k", "vdb_top_k", "collection_name", "messages"}
            kwargs = {key: value for key, value in vars(data).items() if key not in excluded_keys}

            docs = UNSTRUCTURED_RAG.document_search(content=data.query, messages=data.messages, reranker_top_k=data.reranker_top_k, vdb_top_k=data.vdb_top_k, collection_name=data.collection_name, **kwargs)
            citations = prepare_citations(
                collection_name=data.collection_name,
                retrieved_documents=docs,
                force_citations=True
            )
            return citations
        raise NotImplementedError("UnstructuredRAG class has not implemented the document_search method.")

    except asyncio.CancelledError as e:
        logger.warning(f"Request cancelled during document search. {str(e)}")
        return JSONResponse(content={"message": "Request was cancelled by the client."}, status_code=499)
    except Exception as e:
        logger.error("Error from DELETE /documents endpoint. Error details: %s", e)
        return JSONResponse(content={"message": f"Error deleting document {filename}"}, status_code=500)


@app.get("/collections", tags=["Ingestion APIs"], response_model=List[str])
async def get_collections():
    """
    Endpoint to get a list of index names from Pinecone.
    Returns a list of collection names.
    """
    pc = app.state.pinecone
    indexes = pc.list_indexes()
    return [index.name for index in indexes]

def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    logger.info("Shutting down gracefully...")
    try:
        if hasattr(app.state, 'pinecone'):
            # Clean up Pinecone connection if needed
            pass
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Cleaning up resources...")
    try:
        if hasattr(app.state, 'pinecone'):
            # Clean up Pinecone connection if needed
            pass
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.get("/")
async def root():
    return FileResponse("src/static/index.html")
