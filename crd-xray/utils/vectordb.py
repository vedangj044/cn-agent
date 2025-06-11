import logging
import json
from typing import List, Dict, Any, Optional, Union, cast
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from qdrant_client import QdrantClient
from datetime import datetime
import uuid

# Initialize embedding model
logging.info("Initializing embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.info("Embedding model initialized successfully")


# Initialize Qdrant collections
def init_qdrant(qdrant: QdrantClient):
    """Initialize Qdrant collections"""
    logging.info("Initializing Qdrant collections...")
    try:
        existing_collections = {
            collection.name for collection in qdrant.get_collections().collections
        }

        if "crd_data" not in existing_collections:
            logging.info("Creating crd_data collection...")
            qdrant.create_collection(
                collection_name="crd_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info("crd_data collection created successfully")
        else:
            logging.info("crd_data collection already exists")

        if "resource_data" not in existing_collections:
            logging.info("Creating resource_data collection...")
            qdrant.create_collection(
                collection_name="resource_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info("resource_data collection created successfully")
        else:
            logging.info("resource_data collection already exists")

        if "controller_data" not in existing_collections:
            logging.info("Creating controller_data collection...")
            qdrant.create_collection(
                collection_name="controller_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logging.info("controller_data collection created successfully")
        else:
            logging.info("controller_data collection already exists")

    except Exception as e:
        logging.error(f"Failed to initialize Qdrant collections: {str(e)}")
        raise


def create_embedding(text: str) -> List[float]:
    """Create embedding for text using sentence-transformers"""
    return embedding_model.encode(text).tolist()


def store_crd_data(
    crd_name: str, crd_manifest: str, events: List[Dict[str, Any]], qdrant: QdrantClient
):
    """Store CRD data in Qdrant"""

    events_text = "\n".join([json.dumps(event) for event in events])
    combined_text = f"CRD Manifest:\n{crd_manifest}\n\nEvents:\n{events_text}"
    vector = create_embedding(combined_text)

    point_id = str(uuid.uuid4())

    try:
        qdrant.upsert(
            collection_name="crd_data",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "crd_name": crd_name,
                        "page_content": combined_text,
                        "metadata": {
                            "crd_name": crd_name,
                            "timestamp": datetime.now().isoformat(),
                        },
                    },
                )
            ],
        )
        logging.info(f"Successfully stored CRD data for {crd_name} with ID {point_id}")
    except Exception as e:
        logging.error(f"Failed to store CRD data for {crd_name}: {str(e)}")


def store_resource_data(
    resource_name: str,
    crd_name: str,
    resource_manifest: str,
    qdrant: QdrantClient,
):
    """Store resource data in Qdrant"""
    combined_text = f"Resource Manifest:\n{resource_manifest}"
    vector = create_embedding(combined_text)

    point_id = abs(hash(f"{crd_name}_{resource_name}")) % (2**63)

    try:
        qdrant.upsert(
            collection_name="resource_data",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "resource_name": resource_name,
                        "crd_name": crd_name,
                        "page_content": combined_text,
                        "metadata": {
                            "crd_name": crd_name,
                            "resource_name": resource_name,
                            "timestamp": datetime.now().isoformat(),
                        },
                    },
                )
            ],
        )
        logging.info(
            f"Successfully stored resource data for {resource_name} with ID {point_id}"
        )
    except Exception as e:
        logging.error(f"Failed to store resource data for {resource_name}: {str(e)}")


def store_controller_data(
    controller_name: str, controller_manifest: str, logs: str, qdrant: QdrantClient
):
    """Store controller data in Qdrant"""

    combined_text = f"Controller Manifest:\n{controller_manifest}\n\nLogs:\n{logs}"
    vector = create_embedding(combined_text)

    point_id = abs(hash(controller_name)) % (2**63)

    try:
        qdrant.upsert(
            collection_name="controller_data",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "controller_name": controller_name,
                        "page_content": combined_text,
                        "metadata": {
                            "controller_name": controller_name,
                            "timestamp": datetime.now().isoformat(),
                        },
                    },
                )
            ],
        )
        logging.info(
            f"Successfully stored controller data for {controller_name} with ID {point_id}"
        )
    except Exception as e:
        logging.error(
            f"Failed to store controller data for {controller_name}: {str(e)}"
        )


def query_vector_db(
    query: str, collection_name: str, qdrant: QdrantClient, limit: int = 5
) -> List[Dict[str, Any]]:
    """Query Qdrant vector database"""

    query_vector = create_embedding(query)
    search_result = qdrant.search(
        collection_name=collection_name, query_vector=query_vector, limit=limit
    )

    return [hit.payload for hit in search_result if hit.payload is not None]


def delete_qdrant_collections(qdrant: QdrantClient):
    """Delete existing Qdrant collections"""

    logging.info("Deleting existing Qdrant collections...")
    try:
        collections = ["crd_data", "resource_data", "controller_data"]
        for collection in collections:
            try:
                qdrant.delete_collection(collection_name=collection)
                logging.info(f"Deleted collection {collection}")
            except Exception as e:
                logging.warning(f"Failed to delete collection {collection}: {str(e)}")
    except Exception as e:
        logging.error(f"Failed to delete collections: {str(e)}")
        raise
