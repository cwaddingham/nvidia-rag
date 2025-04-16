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
"""Utility functions used across different modules of the RAG."""
import logging
import os
from functools import lru_cache
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Any
from typing import Optional

import yaml
import aiohttp
import asyncio
import time
from .pinecone_utils import get_pinecone_client

logger = logging.getLogger(__name__)

try:
    from nv_ingest_client.client import NvIngestClient, Ingestor
except Exception:
    logger.warning("Optional nv_ingest_client module not installed.")

from src.minio_operator import MinioOperator
import src.configuration as configuration
if TYPE_CHECKING:
    from src.configuration import ConfigWizard

DEFAULT_MAX_CONTEXT = 1500
ENABLE_NV_INGEST_VDB_UPLOAD = True # When enabled entire ingestion would be performed using nv-ingest

# pylint: disable=unnecessary-lambda-assignment

def get_env_variable(
        variable_name: str,
        default_value: Any
    ) -> Any:
    """
    Get an environment variable with a fallback to a default value.
    Also checks if the variable is set, is not empty, and is not longer than 256 characters.
    
    Args:
        variable_name (str): The name of the environment variable to get
        
    Returns:
        Any: The value of the environment variable or the default value if the variable is not set
    """
    var = os.environ.get(variable_name)

    # Check if variable is set
    if var is None:
        logger.warning(f"Environment variable {variable_name} is not set. Using default value: {default_value}")
        var = default_value

    # Check min and max length of variable
    if len(var) > 256 or len(var) == 0:
        logger.warning(f"Environment variable {variable_name} is longer than 256 characters or empty. Using default value: {default_value}")
        var = default_value

    return var

def utils_cache(func: Callable) -> Callable:
    """Use this to convert unhashable args to hashable ones"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable args to hashable ones
        args_hashable = tuple(tuple(arg) if isinstance(arg, (list, dict, set)) else arg for arg in args)
        kwargs_hashable = {
            key: tuple(value) if isinstance(value, (list, dict, set)) else value
            for key, value in kwargs.items()
        }
        return func(*args_hashable, **kwargs_hashable)

    return wrapper


# @lru_cache
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    try:
        config = configuration.AppConfig.from_file(config_file)
        if config:
            return config
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
    raise RuntimeError("Unable to find configuration.")

@utils_cache
def get_vectorstore():
    """Get vector store instance"""
    config = get_config()
    if config.vector_store.url is None:
        if config.vector_store.index_name is not None:
            pc = get_pinecone_client()
            index = pc.Index(config.vector_store.index_name)
            config.vector_store.url = index.config.host
        else:
            logger.warning("No vector store index name provided. Using default.")
            return None

    return get_vectorstore(config.vector_store.url)

@lru_cache
def get_prompts() -> Dict:
    """Retrieves prompt configurations from YAML file and return a dict.
    """

    # default config taking from prompt.yaml
    default_config_path = os.path.join(os.environ.get("EXAMPLE_PATH", os.path.dirname(__file__)), "prompt.yaml")
    default_config = {}
    if Path(default_config_path).exists():
        with open(default_config_path, 'r', encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", default_config_path)
            default_config = yaml.safe_load(file)

    config_file = os.environ.get("PROMPT_CONFIG_FILE", "/prompt.yaml")

    config = {}
    if Path(config_file).exists():
        with open(config_file, 'r', encoding="utf-8") as file:
            logger.info("Using prompts config file from: %s", config_file)
            config = yaml.safe_load(file)

    config = _combine_dicts(default_config, config)
    return config


def get_nv_ingest_client():
    """Creates and returns NV-Ingest client"""
    config = get_config()
    client = NvIngestClient(
        message_client_hostname=config.nv_ingest.message_client_hostname,
        message_client_port=config.nv_ingest.message_client_port
    )
    return client

def get_nv_ingest_ingestor(
        nv_ingest_client_instance,
        filepaths: List[str],
        **kwargs
    ):
    """Prepare NV-Ingest ingestor instance"""
    config = get_config()

    logger.info("Preparing NV Ingest Ingestor instance for filepaths: %s", filepaths)
    ingestor = Ingestor(client=nv_ingest_client_instance)
    ingestor = ingestor.files(filepaths)

    # Add extraction task
    extraction_options = kwargs.get("extraction_options", {})
    ingestor = ingestor.extract(
        extract_text=extraction_options.get("extract_text", config.nv_ingest.extract_text),
        extract_tables=extraction_options.get("extract_tables", config.nv_ingest.extract_tables),
        extract_charts=extraction_options.get("extract_charts", config.nv_ingest.extract_charts),
        extract_images=extraction_options.get("extract_images", config.nv_ingest.extract_images),
        extract_method=extraction_options.get("extract_method", config.nv_ingest.extract_method),
        text_depth=extraction_options.get("text_depth", config.nv_ingest.text_depth),
    )

    # Add splitting task
    split_options = kwargs.get("split_options", {})
    ingestor = ingestor.split(
        tokenizer=config.nv_ingest.tokenizer,
        chunk_size=split_options.get("chunk_size", config.nv_ingest.chunk_size),
        chunk_overlap=split_options.get("chunk_overlap", config.nv_ingest.chunk_overlap),
    )

    # Add captioning if needed
    if extraction_options.get("extract_images", config.nv_ingest.extract_images):
        ingestor = ingestor.caption(
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            endpoint_url=config.nv_ingest.caption_endpoint_url,
            model_name=config.nv_ingest.caption_model_name,
        )

    # Add embedding and vector store tasks
    if ENABLE_NV_INGEST_VDB_UPLOAD:
        ingestor = ingestor.embed()
        ingestor = ingestor.vdb_upload(
            collection_name=kwargs.get("collection_name"),
            milvus_uri=kwargs.get("vdb_endpoint", config.vector_store.url),
            minio_endpoint=os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESSKEY"),
            secret_key=os.getenv("MINIO_SECRETKEY"),
            sparse=(config.vector_store.search_type == "hybrid"),
            enable_images=extraction_options.get("extract_images", config.nv_ingest.extract_images),
            recreate=False,
            dense_dim=config.embeddings.dimensions,
            gpu_index=config.vector_store.enable_gpu_index,
            gpu_search=config.vector_store.enable_gpu_search,
        )

    return ingestor

def get_minio_operator():
    """Get MinIO operator instance with validation"""
    # Check if we're running in Docker by looking for container-specific env vars
    in_container = os.getenv("CONTAINER_ENV") == "true"
    
    # Use localhost for local development, container hostname for Docker
    default_endpoint = "minio:9010" if in_container else "localhost:9010"
    endpoint = os.getenv("MINIO_ENDPOINT", default_endpoint)
    access_key = os.getenv("MINIO_ACCESSKEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRETKEY", "minioadmin")

    logger.debug("Initializing MinIO operator with endpoint: %s", endpoint)
    
    try:
        return MinioOperator(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize MinIO operator: {e}")
        if not in_container:
            logger.warning(
                "If running locally, make sure MinIO is running and accessible at %s. "
                "You can start it with: docker run -p 9010:9010 minio/minio server /data", 
                endpoint
            )
        raise

def get_unique_thumbnail_id_collection_prefix(collection_name: str) -> str:
    """Get prefix for all thumbnails in a collection"""
    return f"{collection_name}::"

def get_unique_thumbnail_id_file_name_prefix(collection_name: str, file_name: str) -> str:
    """Get prefix for all thumbnails from a specific file"""
    collection_prefix = get_unique_thumbnail_id_collection_prefix(collection_name)
    return f"{collection_prefix}{file_name}::"

def get_unique_thumbnail_id(
        collection_name: str,
        file_name: str,
        page_number: int,
        location: List[float]
    ) -> str:
    """Generate unique ID for binary content"""
    rounded_bbox = [round(coord, 4) for coord in location]
    prefix = f"{collection_name}::{file_name}::{page_number}"
    return f"{prefix}_" + "_".join(map(str, rounded_bbox))

def _combine_dicts(dict_a, dict_b):
    """Combines two dictionaries recursively, prioritizing values from dict_b.

    Args:
        dict_a: The first dictionary.
        dict_b: The second dictionary.

    Returns:
        A new dictionary with combined key-value pairs.
    """

    combined_dict = dict_a.copy()  # Start with a copy of dict_a

    for key, value_b in dict_b.items():
        if key in combined_dict:
            value_a = combined_dict[key]
            # Remove the special handling for "command"
            if isinstance(value_a, dict) and isinstance(value_b, dict):
                combined_dict[key] = _combine_dicts(value_a, value_b)
            # Otherwise, replace the value from A with the value from B
            else:
                combined_dict[key] = value_b
        else:
            # Add any key not present in A
            combined_dict[key] = value_b

    return combined_dict

async def check_service_health(
    url: str, 
    service_name: str, 
    method: str = "GET", 
    timeout: int = 5,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Check health of a service endpoint asynchronously.
    
    Args:
        url: The endpoint URL to check
        service_name: Name of the service for reporting
        method: HTTP method to use (GET, POST, etc.)
        timeout: Request timeout in seconds
        headers: Optional HTTP headers
        json_data: Optional JSON payload for POST requests
        
    Returns:
        Dictionary with status information
    """
    start_time = time.time()
    status = {
        "service": service_name,
        "url": url,
        "status": "unknown",
        "latency_ms": 0,
        "error": None
    }
    
    if not url:
        status["status"] = "skipped"
        status["error"] = "No URL provided"
        return status
    
    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        async with aiohttp.ClientSession() as session:
            request_kwargs = {
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "headers": headers or {}
            }
            
            if method.upper() == "POST" and json_data:
                request_kwargs["json"] = json_data
            
            async with getattr(session, method.lower())(url, **request_kwargs) as response:
                status["status"] = "healthy" if response.status < 400 else "unhealthy"
                status["http_status"] = response.status
                status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                
    except asyncio.TimeoutError:
        status["status"] = "timeout"
        status["error"] = f"Request timed out after {timeout}s"
    except aiohttp.ClientError as e:
        status["status"] = "error"
        status["error"] = str(e)
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)
    
    return status

async def check_minio_health(endpoint: str, access_key: str, secret_key: str) -> Dict[str, Any]:
    """Check MinIO server health"""
    status = {
        "service": "MinIO",
        "url": endpoint,
        "status": "unknown",
        "error": None
    }
    
    if not endpoint:
        status["status"] = "skipped"
        status["error"] = "No endpoint provided"
        return status
        
    try:
        start_time = time.time()
        minio_operator = MinioOperator(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key
        )
        # Test basic operation - list buckets
        buckets = minio_operator.client.list_buckets()
        status["status"] = "healthy"
        status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        status["buckets"] = len(buckets)
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)
        
    return status

async def check_pinecone_health(url: str) -> Dict[str, Any]:
    """Check Pinecone database health"""
    status = {
        "service": "Pinecone",
        "url": url,
        "status": "unknown",
        "error": None
    }
    
    try:
        start_time = time.time()
        pc = get_pinecone_client()
        # Test basic operation - list indexes
        indexes = pc.list_indexes()
        
        status["status"] = "healthy"
        status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        status["indexes"] = len(indexes)
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)
        
    return status

async def check_all_services_health() -> Dict[str, List[Dict[str, Any]]]:
    """
    Check health of all services used by the application
    
    Returns:
        Dictionary with service categories and their health status
    """
    config = get_config()
    
    # Create tasks for different service types
    tasks = []
    results = {
        "databases": [],
        "object_storage": [],
        "nim": [],  # New unified category for NIM services
    }
    
    # MinIO health check
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "")
    minio_access_key = os.environ.get("MINIO_ACCESSKEY", "")
    minio_secret_key = os.environ.get("MINIO_SECRETKEY", "")
    if minio_endpoint:
        tasks.append(("object_storage", check_minio_health(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key
        )))
    
    # Vector DB (Milvus) health check
    if config.vector_store.url:
        tasks.append(("databases", check_pinecone_health(config.vector_store.url)))
    
    # LLM service health check
    if config.llm.server_url:
        llm_url = config.llm.server_url
        if not llm_url.startswith(('http://', 'https://')):
            llm_url = f"http://{llm_url}/v1/health/ready"
        else:
            llm_url = f"{llm_url}/v1/health/ready"
        tasks.append(("nim", check_service_health(
            url=llm_url,
            service_name=f"LLM ({config.llm.model_name})"
        )))
    else:
        # When URL is empty, assume the service is running via API catalog
        results["nim"].append({
            "service": f"LLM ({config.llm.model_name})",
            "url": "NVIDIA API Catalog",
            "status": "healthy",
            "latency_ms": 0,
            "message": "Using NVIDIA API Catalog"
        })
    
    query_rewriter_enabled = os.getenv('ENABLE_QUERYREWRITER', 'True').lower() == 'true'

    if query_rewriter_enabled:
        # Query rewriter LLM health check
        if config.query_rewriter.server_url:
            qr_url = config.query_rewriter.server_url
            if not qr_url.startswith(('http://', 'https://')):
                qr_url = f"http://{qr_url}/v1/health/ready"
            else:
                qr_url = f"{qr_url}/v1/health/ready"
            tasks.append(("nim", check_service_health(
                url=qr_url,
                service_name=f"Query Rewriter ({config.query_rewriter.model_name})"
            )))
        else:
            # When URL is empty, assume the service is running via API catalog
            results["nim"].append({
                "service": f"Query Rewriter ({config.query_rewriter.model_name})",
                "url": "NVIDIA API Catalog",
                "status": "healthy",
                "latency_ms": 0,
                "message": "Using NVIDIA API Catalog"
            })
    
    # Embedding service health check
    if config.embeddings.server_url:
        embed_url = config.embeddings.server_url
        if not embed_url.startswith(('http://', 'https://')):
            embed_url = f"http://{embed_url}/v1/health/ready"
        else:
            embed_url = f"{embed_url}/v1/health/ready"
        tasks.append(("nim", check_service_health(
            url=embed_url,
            service_name=f"Embeddings ({config.embeddings.model_name})"
        )))
    else:
        # When URL is empty, assume the service is running via API catalog
        results["nim"].append({
            "service": f"Embeddings ({config.embeddings.model_name})",
            "url": "NVIDIA API Catalog",
            "status": "healthy",
            "latency_ms": 0,
            "message": "Using NVIDIA API Catalog"
        })
    
    enable_reranker = os.getenv('ENABLE_RERANKER', 'True').lower() == 'true'
    # Ranking service health check
    if enable_reranker:
        if config.ranking.server_url:
            ranking_url = config.ranking.server_url
            if not ranking_url.startswith(('http://', 'https://')):
                ranking_url = f"http://{ranking_url}/v1/health/ready"
            else:
                ranking_url = f"{ranking_url}/v1/health/ready"
            tasks.append(("nim", check_service_health(
                url=ranking_url,
                service_name=f"Ranking ({config.ranking.model_name})"
            )))
        else:
            # When URL is empty, assume the service is running via API catalog
            results["nim"].append({
                "service": f"Ranking ({config.ranking.model_name})",
                "url": "NVIDIA API Catalog",
                "status": "healthy",
                "latency_ms": 0,
                "message": "Using NVIDIA API Catalog"
            })
    
    # NemoGuardrails health check
    enable_guardrails = os.getenv('ENABLE_GUARDRAILS', 'False').lower() == 'true'
    if enable_guardrails:
        guardrails_url = os.getenv('NEMO_GUARDRAILS_URL', '')
        if guardrails_url:
            if not guardrails_url.startswith(('http://', 'https://')):
                guardrails_url = f"http://{guardrails_url}/v1/health"
            else:
                guardrails_url = f"{guardrails_url}/v1/health"
            tasks.append(("nim", check_service_health(
                url=guardrails_url,
                service_name="NemoGuardrails"
            )))
        else:
            results["nim"].append({
                "service": "NemoGuardrails",
                "url": "Not configured",
                "status": "skipped",
                "message": "URL not provided"
            })
    
    # Reflection LLM health check
    enable_reflection = os.getenv('ENABLE_REFLECTION', 'False').lower() == 'true'
    if enable_reflection:
        reflection_llm = os.getenv('REFLECTION_LLM', '').strip('"').strip("'")
        reflection_url = os.getenv('REFLECTION_LLM_SERVERURL', '').strip('"').strip("'")
        if reflection_url:
            if not reflection_url.startswith(('http://', 'https://')):
                reflection_url = f"http://{reflection_url}/v1/health/ready"
            else:
                reflection_url = f"{reflection_url}/v1/health/ready"
            tasks.append(("nim", check_service_health(
                url=reflection_url,
                service_name=f"Reflection LLM ({reflection_llm})"
            )))
        else:
            # When URL is empty, assume the service is running via API catalog
            results["nim"].append({
                "service": f"Reflection LLM ({reflection_llm})",
                "url": "NVIDIA API Catalog",
                "status": "healthy",
                "latency_ms": 0,
                "message": "Using NVIDIA API Catalog"
            })
    
    # Execute all health checks concurrently
    for category, task in tasks:
        result = await task
        results[category].append(result)
    
    return results

def print_health_report(health_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Print health status for individual services
    
    Args:
        health_results: Results from check_all_services_health
    """
    logger.info("===== SERVICE HEALTH STATUS =====")
    
    for category, services in health_results.items():
        if not services:
            continue
            
        for service in services:
            if service["status"] == "healthy":
                logger.info(f"Service '{service['service']}' is healthy - Response time: {service.get('latency_ms', 'N/A')}ms")
            elif service["status"] == "skipped":
                logger.info(f"Service '{service['service']}' check skipped - Reason: {service.get('error', 'No URL provided')}")
            else:
                error_msg = service.get("error", "Unknown error")
                logger.info(f"Service '{service['service']}' is not healthy - Issue: {error_msg}")
    
    logger.info("================================")

async def check_and_print_services_health():
    """
    Check health of all services and print a report
    """
    health_results = await check_all_services_health()
    print_health_report(health_results)
    return health_results

def check_services_health():
    """
    Synchronous wrapper for checking service health
    """
    return asyncio.run(check_and_print_services_health())

def get_embedding_model(model: str, url: str = None) -> Any:
    """Get the embedding model based on configuration"""
    config = get_config()
    if config.embeddings.model_engine == "pinecone":
        from .pinecone_utils import PineconeEmbedder
        return PineconeEmbedder(
            model_name=config.embeddings.model_name,
            input_type=config.embeddings.input_type,
            truncate=config.embeddings.truncate
        )
    else:
        raise ValueError(f"Unsupported embedding model engine: {config.embeddings.model_engine}")
