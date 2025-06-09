import kopf
import kubernetes
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import qdrant_client
from qdrant_client.http import models
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load Kubernetes configuration
try:
    kubernetes.config.load_kube_config()
    logger.info("Successfully loaded Kubernetes configuration")
except Exception as e:
    logger.error(f"Failed to load Kubernetes configuration: {str(e)}")
    raise

# Initialize embedding model
logger.info("Initializing embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
    url=os.getenv("QDRANT_URL", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)
logger.info("Qdrant client initialized successfully")

# Initialize SQLite database
def init_db():
    """Initialize SQLite database"""
    logger.info("Initializing SQLite database...")
    conn = sqlite3.connect('k8s_analysis.db')
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {table[0] for table in cursor.fetchall()}
    
    # Create CRD table if not exists
    if 'crd_table' not in existing_tables:
        logger.info("Creating crd_table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS crd_table (
            crd TEXT PRIMARY KEY,
            last_updated_timestamp DATETIME,
            controller_name TEXT,
            names TEXT
        )
        ''')
        logger.info("crd_table created successfully")
    else:
        logger.info("crd_table already exists")
    
    # Create controller table if not exists
    if 'controller_table' not in existing_tables:
        logger.info("Creating controller_table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS controller_table (
            controller TEXT PRIMARY KEY,
            last_updated_timestamp DATETIME
        )
        ''')
        logger.info("controller_table created successfully")
    else:
        logger.info("controller_table already exists")
    
    # Create instance table if not exists
    if 'instance_table' not in existing_tables:
        logger.info("Creating instance_table...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS instance_table (
            resource_name TEXT PRIMARY KEY,
            crd TEXT,
            FOREIGN KEY (crd) REFERENCES crd_table(crd)
        )
        ''')
        logger.info("instance_table created successfully")
    else:
        logger.info("instance_table already exists")
    
    conn.commit()
    return conn

# Initialize Qdrant collections
def init_qdrant():
    """Initialize Qdrant collections"""
    logger.info("Initializing Qdrant collections...")
    try:
        # Get existing collections
        existing_collections = {collection.name for collection in qdrant.get_collections().collections}
        
        # Create or use crd_data collection
        if "crd_data" not in existing_collections:
            logger.info("Creating crd_data collection...")
            qdrant.create_collection(
                collection_name="crd_data",
                vectors_config=models.VectorParams(
                    size=384,  # Updated to match all-MiniLM-L6-v2 model dimension
                    distance=models.Distance.COSINE
                )
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
                    size=384,  # Updated to match all-MiniLM-L6-v2 model dimension
                    distance=models.Distance.COSINE
                )
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
                    size=384,  # Updated to match all-MiniLM-L6-v2 model dimension
                    distance=models.Distance.COSINE
                )
            )
            logger.info("controller_data collection created successfully")
        else:
            logger.info("controller_data collection already exists")
            
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collections: {str(e)}")
        raise

def get_all_crds():
    """Fetch all CRDs from the cluster"""
    logger.info("Fetching all CRDs from the cluster...")
    try:
        crds = custom_api.list_cluster_custom_object(
            group="apiextensions.k8s.io",
            version="v1",
            plural="customresourcedefinitions"
        )
        logger.info(f"Found {len(crds['items'])} CRDs")
        return crds['items']
    except Exception as e:
        logger.error(f"Failed to fetch CRDs: {str(e)}")
        raise

def get_crd_events(crd_name: str):
    """Fetch events related to a specific CRD"""
    logger.info(f"Fetching events for CRD: {crd_name}")
    try:
        events = core_api.list_event_for_all_namespaces(
            field_selector=f"involvedObject.kind=CustomResourceDefinition,involvedObject.name={crd_name}"
        )
        logger.info(f"Found {len(events.items)} events for CRD {crd_name}")
        return events.items
    except Exception as e:
        logger.error(f"Failed to fetch events for CRD {crd_name}: {str(e)}")
        raise

def get_crd_resources(crd: Dict[str, Any]):
    """Fetch all resources for a specific CRD"""
    crd_name = crd['metadata']['name']
    logger.info(f"Fetching resources for CRD: {crd_name}")
    try:
        group = crd['spec']['group']
        version = crd['spec']['versions'][0]['name']
        plural = crd['spec']['names']['plural']
        
        resources = custom_api.list_cluster_custom_object(
            group=group,
            version=version,
            plural=plural
        )
        logger.info(f"Found {len(resources['items'])} resources for CRD {crd_name}")
        return resources['items']
    except Exception as e:
        logger.error(f"Failed to fetch resources for CRD {crd_name}: {str(e)}")
        raise

def get_pod_logs(pod_name: str, namespace: str):
    """Fetch logs for a specific pod"""
    logger.info(f"Fetching logs for pod {pod_name} in namespace {namespace}")
    try:
        logs = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace
        )
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
        creation_timestamp=pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else ""
    )

def identify_controller(crd_name: str, crd_manifest: str, pods: List[Any]) -> str:
    """Identify the controller pod for a CRD using LLM"""
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Convert pods to structured format
    pod_infos = [extract_pod_info(pod) for pod in pods]
    
    prompt = f"""
    Given the following CRD manifest and list of pods, identify which pod is most likely to be the controller for this CRD.
    Return the response in the following JSON format (without any markdown formatting or code blocks):
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
        # Clean the response content
        content = response.content.strip()
        # Remove markdown code block if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        content = content.strip()
        
        result = json.loads(content)
        logger.info(f"Parsed controller identification result for {crd_name}: {json.dumps(result, indent=2)}")
        return result["controller_pod"]
    except Exception as e:
        logger.error(f"Failed to parse controller identification response for {crd_name}: {str(e)}")
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
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
        logger.info(f"Successfully stored CRD data for {crd_name} with ID {point_id}")
    except Exception as e:
        logger.error(f"Failed to store CRD data for {crd_name}: {str(e)}")

def store_resource_data(resource_name: str, crd_name: str, resource_manifest: str, logs: str):
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
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
        logger.info(f"Successfully stored resource data for {resource_name} with ID {point_id}")
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
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
        logger.info(f"Successfully stored controller data for {controller_name} with ID {point_id}")
    except Exception as e:
        logger.error(f"Failed to store controller data for {controller_name}: {str(e)}")

def query_vector_db(query: str, collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Query Qdrant vector database"""
    # Create embedding for query
    query_vector = create_embedding(query)
    
    # Search in Qdrant
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )
    
    return [hit.payload for hit in search_result]

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
    conn = init_db()
    delete_qdrant_collections()  # Delete existing collections
    init_qdrant()  # Reinitialize with correct dimensions
    
    # Collect all CRDs and their resources
    crds = get_all_crds()
    for crd in crds:
        crd_name = crd['metadata']['name']
        crd_manifest = json.dumps(crd)
        crd_events = get_crd_events(crd_name)
        
        # Get all pods
        pods = core_api.list_pod_for_all_namespaces().items
        
        # Identify controller
        controller_pod = identify_controller(crd_name, crd_manifest, pods)
        
        # Update database
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO crd_table (crd, last_updated_timestamp, controller_name, names)
        VALUES (?, ?, ?, ?)
        ''', (crd_name, datetime.now(), controller_pod, json.dumps(crd['spec']['names'])))
        
        if controller_pod:
            cursor.execute('''
            INSERT OR REPLACE INTO controller_table (controller, last_updated_timestamp)
            VALUES (?, ?)
            ''', (controller_pod, datetime.now()))
            
            # Get controller pod info
            controller_pod_obj = next((p for p in pods if p.metadata.name == controller_pod), None)
            if controller_pod_obj:
                pod_info = extract_pod_info(controller_pod_obj)
                controller_manifest = json.dumps(pod_info.__dict__)
                controller_logs = get_pod_logs(controller_pod, pod_info.namespace)
                store_controller_data(controller_pod, controller_manifest, controller_logs)
        
        # Get and store resources
        resources = get_crd_resources(crd)
        for resource in resources:
            resource_name = resource['metadata']['name']
            cursor.execute('''
            INSERT OR REPLACE INTO instance_table (resource_name, crd)
            VALUES (?, ?)
            ''', (resource_name, crd_name))
            
            # Store resource data in Qdrant
            resource_manifest = json.dumps(resource)
            resource_logs = ""  # You might want to implement a way to get logs for custom resources
            store_resource_data(resource_name, crd_name, resource_manifest, resource_logs)
        
        # Store CRD data in Qdrant
        store_crd_data(crd_name, crd_manifest, crd_events)
        
        conn.commit()
    
    conn.close()

@kopf.on.cleanup()
def cleanup_fn(logger, **kwargs):
    """Cleanup resources on shutdown"""
    pass

def process_user_query(query: str, conn: sqlite3.Connection):
    """Process user query using LangGraph and LLM"""
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Define the graph nodes
    def generate_sql(state):
        prompt = f"""
        Convert the following user query into an SQL query for our k8s_analysis database:
        {state['query']}
        
        Available tables:
        - crd_table (crd, last_updated_timestamp, controller_name, names)
        - controller_table (controller, last_updated_timestamp)
        - instance_table (resource_name, crd)
        """
        response = llm.invoke(prompt)
        state['sql_query'] = response.content.strip()
        return state
    
    def execute_sql(state):
        cursor = conn.cursor()
        cursor.execute(state['sql_query'])
        results = cursor.fetchall()
        state['sql_results'] = results
        return state
    
    def query_vector_db(state):
        # Query all three collections in Qdrant
        crd_results = query_vector_db(state['query'], "crd_data")
        resource_results = query_vector_db(state['query'], "resource_data")
        controller_results = query_vector_db(state['query'], "controller_data")
        
        state['vector_results'] = {
            'crd_data': crd_results,
            'resource_data': resource_results,
            'controller_data': controller_results
        }
        return state
    
    def generate_response(state):
        prompt = f"""
        Based on the following information, answer the user's query:
        
        User Query: {state['query']}
        Database Results: {state['sql_results']}
        Vector Search Results: {json.dumps(state['vector_results'], indent=2)}
        """
        response = llm.invoke(prompt)
        state['final_response'] = response.content
        return state
    
    # Create the graph
    workflow = StateGraph(name="query_processor")
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("query_vector_db", query_vector_db)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "query_vector_db")
    workflow.add_edge("query_vector_db", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Compile and run
    app = workflow.compile()
    result = app.invoke({"query": query})
    return result['final_response']

if __name__ == "__main__":
    kopf.run()
