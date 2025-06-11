import kopf
import kubernetes
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, cast
import qdrant_client
from qdrant_client.http import models
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    ForeignKey,
    JSON,
    or_,
    and_,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import expression
from sqlalchemy.orm.attributes import flag_modified


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
Base = declarative_base()


class CRD(Base):
    __tablename__ = "crd_table"

    crd = Column(String, primary_key=True)
    last_updated_timestamp = Column(DateTime)
    controller_name = Column(String)
    names = Column(JSON)

    # Relationship with instances
    instances = relationship("Instance", back_populates="crd")

class Instance(Base):
    __tablename__ = "instance_table"

    resource_name = Column(String, primary_key=True)
    crd = Column(String, ForeignKey("crd_table.crd"))

    # Relationship with CRD
    crd_obj = relationship("CRD", back_populates="instances")


# Create engine and session
engine = create_engine("sqlite:///k8s_analysis.db")
Session = sessionmaker(bind=engine)


# Initialize database
def init_db():
    """Initialize SQLAlchemy database"""
    logger.info("Initializing database...")
    Base.metadata.create_all(engine)
    logger.info("Database initialized successfully")


# Initialize Qdrant collections
def init_qdrant():
    """Initialize Qdrant collections"""
    logger.info("Initializing Qdrant collections...")
    try:
        # Get existing collections
        existing_collections = {
            collection.name for collection in qdrant.get_collections().collections
        }

        # Create or use crd_data collection
        if "crd_data" not in existing_collections:
            logger.info("Creating crd_data collection...")
            qdrant.create_collection(
                collection_name="crd_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("crd_data collection created successfully")
        else:
            logger.info("crd_data collection already exists")

        # Create or use resource_data collection
        if "resource_data" not in existing_collections:
            logger.info("Creating resource_data collection...")
            qdrant.create_collection(
                collection_name="resource_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("resource_data collection created successfully")
        else:
            logger.info("resource_data collection already exists")

        # Create or use controller_data collection
        if "controller_data" not in existing_collections:
            logger.info("Creating controller_data collection...")
            qdrant.create_collection(
                collection_name="controller_data",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("controller_data collection created successfully")
        else:
            logger.info("controller_data collection already exists")

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collections: {str(e)}")
        raise


# Load Kubernetes configuration
try:
    kubernetes.config.load_kube_config()
    logger.info("Successfully loaded Kubernetes configuration")
except Exception as e:
    logger.error(f"Failed to load Kubernetes configuration: {str(e)}")
    raise

# Initialize embedding model
logger.info("Initializing embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model initialized successfully")

# Initialize Kubernetes client
logger.info("Initializing Kubernetes client...")
k8s_client = kubernetes.client.ApiClient()
custom_api = kubernetes.client.CustomObjectsApi(k8s_client)
core_api = kubernetes.client.CoreV1Api(k8s_client)
logger.info("Kubernetes client initialized successfully")

# Initialize Qdrant client
logger.info("Initializing Qdrant client...")
qdrant = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333))
)
logger.info("Qdrant client initialized successfully")

MODEL_NAME = "claude-3-7-sonnet-20250219"


def get_all_crds():
    """Fetch all CRDs from the cluster"""
    logger.info("Fetching all CRDs from the cluster...")
    try:
        crds = custom_api.list_cluster_custom_object(
            group="apiextensions.k8s.io",
            version="v1",
            plural="customresourcedefinitions",
        )
        logger.info(f"Found {len(crds['items'])} CRDs")
        return crds["items"]
    except Exception as e:
        logger.error(f"Failed to fetch CRDs: {str(e)}")
        raise


def get_crd_events(crd_kind: str):
    """Fetch events related to a specific CRD"""
    logger.info(f"Fetching events for CRD: {crd_kind}")
    try:
        events = core_api.list_event_for_all_namespaces(
            field_selector=f"involvedObject.kind={crd_kind}"
        )
        logger.info(f"Found {len(events.items)} events for CRD {crd_kind}")
        return events.items
    except Exception as e:
        logger.error(f"Failed to fetch events for CRD {crd_kind}: {str(e)}")
        raise


def get_latest_last_transition_time(resource: dict) -> Optional[str]:
    """Fetch latest timestamp from status of the CRD"""
    try:
        conditions = resource.get("status", {}).get("conditions", [])
        timestamps = [
            cond["lastTransitionTime"]
            for cond in conditions
            if "lastTransitionTime" in cond
        ]

        if not timestamps:
            return None

        latest = max(
            datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps
        )
        return latest.isoformat()

    except Exception as e:
        logger.error(f"Failed to extract lastTransitionTime: {e}")
        return None


def get_crd_resources(crd: Dict[str, Any]):
    """Fetch all resources for a specific CRD"""
    crd_name = crd["metadata"]["name"]
    logger.info(f"Fetching resources for CRD: {crd_name}")
    try:
        group = crd["spec"]["group"]
        version = crd["spec"]["versions"][0]["name"]
        plural = crd["spec"]["names"]["plural"]

        resources = custom_api.list_cluster_custom_object(
            group=group, version=version, plural=plural
        )
        logger.info(f"Found {len(resources['items'])} resources for CRD {crd_name}")
        return resources["items"]
    except Exception as e:
        logger.error(f"Failed to fetch resources for CRD {crd_name}: {str(e)}")
        raise


def get_pod_logs(pod_name: str, namespace: str):
    """Fetch logs for a specific pod"""
    logger.info(f"Fetching logs for pod {pod_name} in namespace {namespace}")
    try:
        logs = core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace)
        logger.info(f"Successfully fetched logs for pod {pod_name}")
        return logs
    except Exception as e:
        logger.error(f"Error fetching logs for pod {pod_name}: {str(e)}")
        return ""


@dataclass
class PodInfo:
    name: str
    namespace: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    container_images: List[str]
    status: str
    creation_timestamp: str


def extract_pod_info(pod) -> PodInfo:
    """Extract relevant information from a Kubernetes pod object"""
    return PodInfo(
        name=pod.metadata.name,
        namespace=pod.metadata.namespace,
        labels=pod.metadata.labels or {},
        annotations=pod.metadata.annotations or {},
        container_images=[container.image for container in pod.spec.containers],
        status=pod.status.phase,
        creation_timestamp=(
            pod.metadata.creation_timestamp.isoformat()
            if pod.metadata.creation_timestamp
            else ""
        ),
    )


def identify_controller(crd_name: str, crd_manifest: str, pods: List[Any]) -> str:
    """Identify the controller pod for a CRD using LLM"""
    llm = ChatAnthropic(
        model_name=MODEL_NAME,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=60,
        stop=None,
        temperature=0,
    )

    # Convert pods to structured format
    pod_infos = [extract_pod_info(pod) for pod in pods]

    prompt = f"""
    Given the following CRD manifest and list of pods, identify which pod is most likely to be the controller for this CRD.
    Return the response in the following JSON format (do not include markdown formatting or code blocks):
    {{
        "controller_pod": "pod_name",
        "confidence": 0.95,
        "reasoning": "explanation of why this pod is likely the controller"
    }}
    
    CRD Name: {crd_name}
    CRD Manifest: {crd_manifest}
    
    Available Pods:
    {json.dumps([pod.__dict__ for pod in pod_infos], indent=2)}
    """

    logger.info(f"Sending prompt to Claude for CRD {crd_name}")
    response = llm.invoke(prompt)
    logger.info(f"Raw Claude response for {crd_name}: {response.content}")

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        content = content.strip()

        result = json.loads(content)
        logger.info(
            f"Parsed controller identification result for {crd_name}: {json.dumps(result, indent=2)}"
        )
        return result["controller_pod"]
    except Exception as e:
        logger.error(
            f"Failed to parse controller identification response for {crd_name}: {str(e)}"
        )
        logger.error(f"Cleaned content that failed to parse: {content}")
        return ""


def create_embedding(text: str) -> List[float]:
    """Create embedding for text using sentence-transformers"""
    return embedding_model.encode(text).tolist()


def store_crd_data(crd_name: str, crd_manifest: str, events: List[Dict[str, Any]]):
    """Store CRD data in Qdrant"""
    # Combine CRD manifest and events into a single text
    events_text = "\n".join([json.dumps(event) for event in events])
    combined_text = f"CRD Manifest:\n{crd_manifest}\n\nEvents:\n{events_text}"

    # Create embedding
    vector = create_embedding(combined_text)

    # Generate a positive integer ID from the hash
    point_id = abs(hash(crd_name)) % (2**63)  # Ensure positive 64-bit integer

    # Store in Qdrant
    try:
        qdrant.upsert(
            collection_name="crd_data",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "crd_name": crd_name,
                        "text": combined_text,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ],
        )
        logger.info(f"Successfully stored CRD data for {crd_name} with ID {point_id}")
    except Exception as e:
        logger.error(f"Failed to store CRD data for {crd_name}: {str(e)}")


def store_resource_data(
    resource_name: str, crd_name: str, resource_manifest: str, logs: str
):
    """Store resource data in Qdrant"""
    # Combine resource manifest and logs
    combined_text = f"Resource Manifest:\n{resource_manifest}\n\nLogs:\n{logs}"

    # Create embedding
    vector = create_embedding(combined_text)

    # Generate a positive integer ID from the hash
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
                        "text": combined_text,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ],
        )
        logger.info(
            f"Successfully stored resource data for {resource_name} with ID {point_id}"
        )
    except Exception as e:
        logger.error(f"Failed to store resource data for {resource_name}: {str(e)}")


def store_controller_data(controller_name: str, controller_manifest: str, logs: str):
    """Store controller data in Qdrant"""
    # Combine controller manifest and logs
    combined_text = f"Controller Manifest:\n{controller_manifest}\n\nLogs:\n{logs}"

    # Create embedding
    vector = create_embedding(combined_text)

    # Generate a positive integer ID from the hash
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
                        "text": combined_text,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ],
        )
        logger.info(
            f"Successfully stored controller data for {controller_name} with ID {point_id}"
        )
    except Exception as e:
        logger.error(f"Failed to store controller data for {controller_name}: {str(e)}")


def query_vector_db(
    query: str, collection_name: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Query Qdrant vector database"""
    # Create embedding for query
    query_vector = create_embedding(query)

    # Search in Qdrant
    search_result = qdrant.search(
        collection_name=collection_name, query_vector=query_vector, limit=limit
    )

    return [hit.payload for hit in search_result if hit.payload is not None]


def delete_qdrant_collections():
    """Delete existing Qdrant collections"""
    logger.info("Deleting existing Qdrant collections...")
    try:
        collections = ["crd_data", "resource_data", "controller_data"]
        for collection in collections:
            try:
                qdrant.delete_collection(collection_name=collection)
                logger.info(f"Deleted collection {collection}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to delete collections: {str(e)}")
        raise


@kopf.on.startup()
def startup_fn(logger, **kwargs):
    """Initialize database and Qdrant on startup"""
    init_db()
    init_qdrant()


@kopf.on.cleanup()
def cleanup_fn(logger, **kwargs):
    """Cleanup resources on shutdown"""
    pass


@kopf.timer(
    "v1",
    "configmap",
    interval=60.0,
    id="periodic-update",
    annotations={"kopf.io/ensure-single-instance": "true"},
)
def periodic_update(logger, **kwargs):
    """Periodically update CRD and resource data every 60 seconds"""

    # Periodically check for CRD
    # check if the manifest has changes, if yes update the manifest and update the timestamp
    # check if the controller has changed, if yes update the controller
    # check if there are new events, if yes update the text and update the timestamp
    # check if there are new resources, if yes update the db and extract the text and manifest 
    # 
    # Less frequent
    # get new logs from controller pod and generate a lifecycle statement from it.

    logger.info("Starting periodic update...")

    # Collect all CRDs and their resources
    crds = get_all_crds()
    session = Session()
    try:
        for crd in crds:
            crd_name = crd["metadata"]["name"]
            crd_kind = crd["spec"]["names"]["kind"]
            crd_manifest = json.dumps(crd)
            manifest_update_required = False

            # Check if CRD exists in database and compare timestamps
            crd_obj = session.query(CRD).filter(CRD.crd == crd_name).first()
            if not crd_obj:
                crd_obj = CRD(crd=crd_name)

            current_latest_timestamp = get_latest_last_transition_time(crd)
            timestamp_value = getattr(crd_obj, "last_updated_timestamp", None)
            if (timestamp_value is None or timestamp_value != current_latest_timestamp):
                manifest_update_required = True

            # Get all pods
            pods = core_api.list_pod_for_all_namespaces().items

            # Check if stored controller pod still exists
            stored_controller = getattr(crd_obj, "controller_name", None)
            controller_pod = None

            if stored_controller is not None and stored_controller != "":
                # Check if the stored controller pod still exists
                controller_exists = any(
                    pod.metadata.name == stored_controller for pod in pods
                )
                if controller_exists:
                    logger.info(
                        f"Using existing controller pod {stored_controller} for CRD {crd_name}"
                    )
                    controller_pod = stored_controller
                else:
                    logger.info(
                        f"Stored controller pod {stored_controller} no longer exists for CRD {crd_name}, identifying new controller"
                    )
                    controller_pod = identify_controller(crd_name, crd_manifest, pods)
            else:
                logger.info(
                    f"No stored controller found for CRD {crd_name}, identifying new controller"
                )
                controller_pod = identify_controller(crd_name, crd_manifest, pods)

            # Update database
            if controller_pod is not None:
                setattr(crd_obj, "controller_name", controller_pod)
            setattr(crd_obj, "last_updated_timestamp", current_latest_timestamp)
            setattr(crd_obj, "names", crd["spec"]["names"])
            session.add(crd_obj)

            # Get and store resources
            resources = get_crd_resources(crd)
            for resource in resources:
                resource_name = resource["metadata"]["name"]

                # Check if resource exists
                instance_obj = (
                    session.query(Instance)
                    .filter(Instance.resource_name == resource_name)
                    .first()
                )
                if not instance_obj:
                    instance_obj = Instance(resource_name=resource_name, crd=crd_name)
                    session.add(instance_obj)

                    # Store resource data in Qdrant
                    resource_manifest = json.dumps(resource)
                    # Get existing resource data from Qdrant
                    existing_resource_data = query_vector_db(
                        resource_name, "resource_data", limit=1
                    )
                    existing_logs = (
                        existing_resource_data[0]
                        .get("text", "")
                        .split("\n\nLogs:\n")[-1]
                        if existing_resource_data
                        else ""
                    )
                    store_resource_data(
                        resource_name, crd_name, resource_manifest, existing_logs
                    )

            # Store CRD data in Qdrant
            # Get existing CRD data from Qdrant
            existing_crd_data = query_vector_db(crd_name, "crd_data", limit=1)
            existing_events = (
                existing_crd_data[0].get("text", "").split("\n\nEvents:\n")[-1]
                if existing_crd_data
                else ""
            )
            store_crd_data(crd_name, crd_manifest, existing_events)

        session.commit()
    finally:
        session.close()
    logger.info("Periodic update completed")


@kopf.timer(
    "v1",
    "configmap",
    interval=3600.0,
    id="log-analysis",
    annotations={"kopf.io/ensure-single-instance": "true"},
)
def analyze_controller_logs(logger, **kwargs):
    """Analyze controller logs every hour to track lifecycle and errors"""
    logger.info("Starting hourly log analysis...")
    session = Session()

    try:
        # Get all CRDs and their controllers
        crd_controllers = session.query(CRD).filter(CRD.controller_name != None).all()

        for crd_obj in crd_controllers:
            logger.info(
                f"Analyzing logs for controller {crd_obj.controller_name} of CRD {crd_obj.crd}"
            )

            # Get current pod status
            pods = core_api.list_pod_for_all_namespaces().items
            controller_pod = next(
                (pod for pod in pods if pod.metadata.name == crd_obj.controller_name),
                None,
            )

            if not controller_pod:
                logger.warning(
                    f"Controller pod {crd_obj.controller_name} no longer exists"
                )
                continue

            # Get current logs (last 100 lines)
            try:
                current_logs = core_api.read_namespaced_pod_log(
                    name=crd_obj.controller_name,
                    namespace=controller_pod.metadata.namespace,
                    tail_lines=100,
                )
            except Exception as e:
                logger.error(
                    f"Failed to fetch logs for pod {crd_obj.controller_name}: {str(e)}"
                )
                continue

            # Get stored logs from vector database
            existing_controller_data = query_vector_db(
                crd_obj.controller_name, "controller_data", limit=1
            )
            stored_logs = (
                existing_controller_data[0].get("text", "")
                if existing_controller_data
                else ""
            )

            # Prepare prompt for LLM
            llm = ChatAnthropic(
                model_name=MODEL_NAME,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                timeout=60,
                stop=None,
                temperature=0,
            )

            prompt = f"""
            Analyze the following controller logs and create a structured lifecycle summary.
            The logs are from a Kubernetes controller pod that manages a Custom Resource Definition (CRD).
            
            Previous logs / lifecycle events:
            {stored_logs}
            
            Current logs:
            {current_logs}
            
            Please analyze these logs and provide a response in the following format:

            lifecycle_events
            <Describe the normal lifecycle events, startup, initialization, processing cycles, etc.>

            error_events
            <Describe any errors, their types, and descriptions>

            Focus on:
            1. Identifying normal lifecycle events (startup, initialization, processing cycles)
            2. Detecting and categorizing errors
            3. Understanding the controller's behavior patterns
            4. Highlighting any anomalies or concerns
            """

            try:
                response = llm.invoke(prompt)
                analysis_text = response.content

                # Create a structured log summary
                log_summary = {
                    "analysis": analysis_text,
                    "last_updated": datetime.now().isoformat(),
                }

                logger.info(f"Log summary: {log_summary}")

                # Update the controller data in vector database with new analysis
                controller_manifest = json.dumps(
                    extract_pod_info(controller_pod).__dict__
                )
                store_controller_data(
                    crd_obj.controller_name,
                    controller_manifest,
                    json.dumps(log_summary, indent=2),
                )

                logger.info(
                    f"Successfully analyzed and updated logs for controller {crd_obj.controller_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to analyze logs for controller {crd_obj.controller_name}: {str(e)}"
                )
                continue

    finally:
        session.close()
        logger.info("Hourly log analysis completed")


if __name__ == "__main__":
    kopf.run()
