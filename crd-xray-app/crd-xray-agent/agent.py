import os
import logging
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
import qdrant_client
from langgraph.graph import StateGraph, END, START
from utils.state import AgentState
from utils.tools import get_sql_tool
from utils.nodes import SQLAgent, VectorAgent, SynthesizerAgent

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "claude-3-7-sonnet-20250219"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///k8s_analysis.db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)

llm = ChatAnthropic(
    model_name=MODEL_NAME,
    api_key=ANTHROPIC_API_KEY,
    timeout=60,
    stop=None,
    temperature=0,
)
logging.info("LLM Model initialized successfully")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logging.info("Embedding model initialized successfully")

engine = create_engine(DATABASE_URL, echo=True, future=True)
sql_db = SQLDatabase(engine)
logging.info("Initialized Database engine")

qdrant = qdrant_client.QdrantClient(url=QDRANT_URL, port=int(QDRANT_PORT))
logging.info("Qdrant client initialized successfully")

workflow = StateGraph(AgentState)

sql_tool = get_sql_tool(sql_db, llm)
sql_agent = SQLAgent(llm, sql_tool, "crd-xray-agent/prompts/sql_prompt.txt")
vector_agent = VectorAgent(qdrant, embedding)
synthesizer = SynthesizerAgent(llm, "crd-xray-agent/prompts/synthesizer_prompt.txt")

workflow.add_node("sql_agent", sql_agent)
workflow.add_node("vector_agent", vector_agent)
workflow.add_node("synthesizer", synthesizer)

workflow.add_edge(START, "sql_agent")
workflow.add_edge("sql_agent", "vector_agent")
workflow.add_edge("vector_agent", "synthesizer")
workflow.add_edge("synthesizer", END)

graph = workflow.compile()
