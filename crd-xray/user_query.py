import sqlite3
import json
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv
import logging
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models
from setup import create_embedding  # Import the embedding function

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def clean_json_response(response_content: str) -> str:
    """Clean JSON response by removing markdown formatting and code blocks."""
    content = response_content.strip()
    
    # Remove markdown code block if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    if content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove ```
    
    return content.strip()

# Data Models
@dataclass
class SQLQuery:
    query: str
    explanation: str

@dataclass
class SQLResult:
    columns: List[str]
    rows: List[tuple]
    error: Optional[str] = None

@dataclass
class VectorSearchResult:
    collection: str
    results: List[Dict[str, Any]]
    score: float

@dataclass
class QueryResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class QueryState(TypedDict):
    query: str
    sql_query: Optional[SQLQuery]
    sql_results: Optional[SQLResult]
    vector_results: Optional[Dict[str, List[VectorSearchResult]]]
    final_response: Optional[QueryResponse]

def process_user_query(query: str, conn: sqlite3.Connection) -> QueryResponse:
    """Process user query using LangGraph and LLM"""
    logger.info(f"Processing user query: {query}")
    
    llm = ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        timeout=60,
        stop=None,
        temperature=0  # Set temperature to 0 for deterministic responses
    )
    logger.info("Initialized ChatAnthropic client")
    
    # Initialize Qdrant client
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )
    logger.info("Initialized Qdrant client")
    
    # Define the graph nodes
    def generate_sql(state: QueryState) -> QueryState:
        logger.info("Generating SQL query from user query")
        prompt = f"""
        Convert the following user query into an SQL query for our k8s_analysis database.
        Return the response in the following JSON format (do not include markdown formatting or code blocks):
        {{
            "query": "SQL query string",
            "explanation": "Explanation of what the query does"
        }}
        
        User Query: {state['query']}
        
        Available tables:
        - crd_table (crd, last_updated_timestamp, controller_name, names)
        - controller_table (controller, last_updated_timestamp)
        - instance_table (resource_name, crd)
        """
        logger.debug(f"SQL generation prompt: {prompt}")
        
        response = llm.invoke(prompt)
        logger.info(f"Raw Claude response for SQL generation: {response.content}")
        
        try:
            cleaned_content = clean_json_response(response.content)
            result = json.loads(cleaned_content)
            state['sql_query'] = SQLQuery(
                query=result["query"],
                explanation=result["explanation"]
            )
            logger.info(f"Successfully generated SQL query: {result['query']}")
            logger.info(f"Query explanation: {result['explanation']}")
        except Exception as e:
            logger.error(f"Failed to parse SQL generation response: {str(e)}")
            logger.error(f"Raw response content: {response.content}")
            state['sql_query'] = SQLQuery(
                query="",
                explanation=f"Error generating SQL: {str(e)}"
            )
        return state
    
    def execute_sql(state: QueryState) -> QueryState:
        logger.info("Executing SQL query")
        if not state['sql_query'] or not state['sql_query'].query:
            logger.warning("No SQL query to execute")
            state['sql_results'] = SQLResult([], [], "No SQL query generated")
            return state
            
        cursor = conn.cursor()
        try:
            logger.debug(f"Executing SQL query: {state['sql_query'].query}")
            cursor.execute(state['sql_query'].query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            state['sql_results'] = SQLResult(columns=columns, rows=rows)
            logger.info(f"SQL query executed successfully. Found {len(rows)} rows")
            logger.debug(f"SQL results - Columns: {columns}")
            logger.debug(f"SQL results - First few rows: {rows[:5]}")
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            state['sql_results'] = SQLResult([], [], str(e))
        return state
    
    def query_vector_db(state: QueryState) -> QueryState:
        logger.info("Querying vector database")
        collections = ["crd_data", "resource_data", "controller_data"]
        vector_results = {}
        
        for collection in collections:
            try:
                logger.info(f"Querying collection: {collection}")
                # Create embedding for query
                query_vector = create_embedding(state['query'])
                logger.debug(f"Created embedding vector of length: {len(query_vector)}")
                
                # Search in Qdrant
                search_result = qdrant.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=5
                )
                
                vector_results[collection] = [
                    VectorSearchResult(
                        collection=collection,
                        results=[hit.payload for hit in search_result],
                        score=hit.score
                    )
                    for hit in search_result
                ]
                logger.info(f"Found {len(search_result)} results in collection {collection}")
                logger.debug(f"Vector search results for {collection}: {json.dumps([hit.payload for hit in search_result], indent=2)}")
            except Exception as e:
                logger.error(f"Error querying vector DB for collection {collection}: {str(e)}")
                vector_results[collection] = []
        
        state['vector_results'] = vector_results
        return state
    
    def generate_response(state: QueryState) -> QueryState:
        logger.info("Generating final response")
        prompt = f"""
        Based on the following information, answer the user's query.
        Return the response in the following JSON format (do not include markdown formatting or code blocks):
        {{
            "answer": "Detailed answer to the user's query",
            "sources": [
                {{
                    "type": "sql|vector",
                    "collection": "collection_name",
                    "data": "relevant data used"
                }}
            ],
            "confidence": 0.95
        }}
        
        User Query: {state['query']}
        
        SQL Results:
        {json.dumps({
            "query": state['sql_query'].query if state['sql_query'] else None,
            "explanation": state['sql_query'].explanation if state['sql_query'] else None,
            "results": {
                "columns": state['sql_results'].columns if state['sql_results'] else [],
                "rows": state['sql_results'].rows if state['sql_results'] else []
            }
        }, indent=2)}
        
        Vector Search Results:
        {json.dumps({
            collection: [
                {
                    "score": result.score,
                    "data": result.results
                }
                for result in results
            ]
            for collection, results in (state['vector_results'] or {}).items()
        }, indent=2)}
        """
        
        logger.debug(f"Response generation prompt: {prompt}")
        response = llm.invoke(prompt)
        logger.info(f"Raw Claude response for final answer: {response.content}")
        
        try:
            cleaned_content = clean_json_response(response.content)
            result = json.loads(cleaned_content)
            state['final_response'] = QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                confidence=result["confidence"]
            )
            logger.info(f"Successfully generated response with confidence: {result['confidence']}")
            logger.debug(f"Final response: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to parse final response: {str(e)}")
            logger.error(f"Raw response content: {response.content}")
            state['final_response'] = QueryResponse(
                answer="Error generating response",
                sources=[],
                confidence=0.0
            )
        return state
    
    # Create the graph
    logger.info("Creating workflow graph")
    workflow = StateGraph(
        state_schema=QueryState
    )
    
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
    
    # Add START edge
    workflow.add_edge(START, "generate_sql")
    
    # Compile and run
    logger.info("Compiling workflow")
    app = workflow.compile()
    initial_state = {
        "query": query,
        "sql_query": None,
        "sql_results": None,
        "vector_results": None,
        "final_response": None
    }
    logger.info("Invoking workflow")
    result = app.invoke(initial_state)
    logger.info("Workflow completed successfully")
    return result['final_response']

def main():
    """Main function to demonstrate query processing"""
    logger.info("Starting query processing demonstration")
    
    # Connect to SQLite database
    conn = sqlite3.connect('k8s_analysis.db')
    logger.info("Connected to SQLite database")
    
    # Sample query
    sample_query = "are there any errors in the logs of the sealed secrets controller?"
    logger.info(f"Processing sample query: {sample_query}")
    
    try:
        # Process the query
        response = process_user_query(sample_query, conn)
        
        # Print results
        logger.info("\nQuery Results:")
        logger.info(f"Query: {sample_query}")
        logger.info(f"Answer: {response.answer}")
        logger.info(f"Confidence: {response.confidence}")
        logger.info("Sources:")
        for source in response.sources:
            logger.info(f"- Type: {source['type']}")
            logger.info(f"  Collection: {source['collection']}")
            logger.info(f"  Data: {json.dumps(source['data'], indent=2)}")
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main()