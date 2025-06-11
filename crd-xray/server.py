import kopf
import kubernetes
import logging
import qdrant_client
import os
from sentence_transformers import SentenceTransformer
from utils.db import init_db, SessionLocal
from utils.k8s import (
    get_all_crds,
    get_crd_events,
    get_crd_resources,
    get_controller_logs,
    get_controller,
)
from utils.ai_preprocessing import identify_controller, summarize_logs
from utils.vectordb import (
    store_crd_data,
    store_resource_data,
    init_qdrant,
    store_controller_data,
    query_vector_db,
)
import utils.db
import utils.vectordb
import json
from models import CRD, Instance
from datetime import datetime, timezone


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load Kubernetes configuration
try:
    kubernetes.config.load_kube_config()
    logging.info("Successfully loaded Kubernetes configuration")
except Exception as e:
    logging.error(f"Failed to load Kubernetes configuration: {str(e)}")
    raise

# Initialize Kubernetes client
logging.info("Initializing Kubernetes client...")
k8s_client = kubernetes.client.ApiClient()
custom_api = kubernetes.client.CustomObjectsApi(k8s_client)
core_api = kubernetes.client.CoreV1Api(k8s_client)
logging.info("Kubernetes client initialized successfully")

# Initialize Qdrant client
logging.info("Initializing Qdrant client...")
qdrant = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333))
)
logging.info("Qdrant client initialized successfully")


MODEL_NAME = "claude-3-7-sonnet-20250219"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///k8s_analysis.db")

if ANTHROPIC_API_KEY is None or DATABASE_URL is None:
    raise EnvironmentError(
        "Required environment variable ANTHROPIC_API_KEY / DATABASE_URL is not set."
    )


@kopf.on.startup()
def startup_fn(**kwargs):
    """Initialize database and Qdrant on startup"""
    init_db()
    init_qdrant(qdrant)


@kopf.on.cleanup()
def cleanup_fn(**kwargs):
    """Cleanup resources on shutdown"""
    pass


@kopf.timer(
    "v1",
    "configmap",
    interval=60.0,
    id="periodic-update",
    annotations={"kopf.io/ensure-single-instance": "true"},
)
def periodic_update(**kwargs):
    """Periodically update CRD and resource data every 60 seconds"""

    logging.info("Starting periodic update...")

    crds = get_all_crds(custom_api)
    session = SessionLocal()
    for crd in crds:
        crd_name = crd["metadata"]["name"]
        crd_kind = crd["spec"]["names"]["kind"]
        crd_names = crd["spec"]["names"]
        crd_manifest = json.dumps(crd)

        pods = core_api.list_pod_for_all_namespaces().items

        crd_obj = session.query(CRD).filter(CRD.crd == crd_name).first()

        if not crd_obj:
            crd_obj = CRD(crd=crd_name)

        if crd_obj.controller_name is None:
            crd_obj.controller_name = identify_controller(
                crd_name, crd_manifest, pods, MODEL_NAME, ANTHROPIC_API_KEY
            )
            logging.info("Controlled updated (prev: NA) - " + crd_obj.controller_name)

        if any(pod.metadata.name == crd_obj.controller_name for pod in pods) is False:
            crd_obj.controller_name = identify_controller(
                crd_name, crd_manifest, pods, MODEL_NAME, ANTHROPIC_API_KEY
            )
            logging.info(
                "Controlled updated: prev pod doesn't exist - "
                + crd_obj.controller_name
            )

        crd_events = get_crd_events(crd_kind, crd_obj.last_updated_timestamp, core_api)
        crd_obj.last_updated_timestamp = datetime.now(timezone.utc)

        if crd_obj.names is None:
            crd_obj.names = crd_names

        session.add(crd_obj)
        if len(crd_events) > 0:
            store_crd_data(crd_name, crd_manifest, crd_events, qdrant)

    for crd in crds:
        resources = get_crd_resources(crd, custom_api)

        for resource in resources:
            resource_name = resource["metadata"]["name"]

            instance_obj = (
                session.query(Instance)
                .filter(Instance.resource_name == resource_name)
                .first()
            )

            if not instance_obj:
                instance_obj = Instance(resource_name=resource_name, crd=crd_name)
                resource_manifest = json.dumps(resource)

                session.add(instance_obj)
                store_resource_data(resource_name, crd_name, resource_manifest, qdrant)

    session.commit()
    session.close()
    logging.info("Periodic update completed")


@kopf.timer(
    "v1",
    "configmap",
    interval=3600.0,
    id="log-analysis",
    annotations={"kopf.io/ensure-single-instance": "true"},
)
def analyze_controller_logs(**kwargs):
    """Analyze controller logs every hour to track lifecycle and errors"""

    logging.info("Starting hourly log analysis...")
    session = SessionLocal()

    crd_controllers = session.query(CRD).filter(CRD.controller_name != None).all()

    for crd_obj in crd_controllers:
        logging.info(
            f"Analyzing logs for controller {crd_obj.controller_name} of CRD {crd_obj.crd}"
        )

        controller_manifest = get_controller(crd_obj, core_api)
        if controller_manifest is None:
            continue

        current_logs = get_controller_logs(
            crd_obj, controller_manifest.metadata.namespace, core_api
        )

        existing_controller_data = query_vector_db(
            crd_obj.controller_name, "controller_data", qdrant, limit=1
        )
        stored_logs = (
            existing_controller_data[0].get("page_content", "").split("\n\nLogs:\n")[-1]
            if existing_controller_data
            else ""
        )

        log_summary = summarize_logs(
            crd_obj.controller_name,
            stored_logs,
            current_logs,
            MODEL_NAME,
            ANTHROPIC_API_KEY,
        )
        if log_summary is None:
            continue

        store_controller_data(
            crd_obj.controller_name,
            json.dumps(controller_manifest.to_dict(), default=str),
            str(log_summary),
            qdrant,
        )

    session.close()
    logging.info("Hourly log analysis completed")
