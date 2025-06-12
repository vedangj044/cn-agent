from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    sql_results: Optional[str]
    vector_results: Optional[str]
    route_decision: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
