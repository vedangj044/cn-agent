import logging
from langgraph.prebuilt import create_react_agent
from .state import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_qdrant import QdrantVectorStore


class SQLAgent:
    def __init__(self, llm, sql_tool, prompt_file):
        self.llm = llm
        prompt = open(prompt_file, "r").read()
        self.sql_agent = create_react_agent(llm, [sql_tool], prompt=prompt)

    def __call__(self, state: AgentState) -> AgentState:
        """SQL agent that uses LangChain SQL tool"""
        query = state.get("query", "").strip()
        
        if query == "" and len(state.get("messages")) == 1:
            last_msg = state.get("messages")[-1]
            state["query"] = last_msg.get("content", "").strip() # type: ignore
            query = state.get("query")
        
        try:
            agent_output = self.sql_agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )

            logging.info(f"SQL Agent executed query successfully")

            if isinstance(agent_output, AIMessage):
                sql_results = agent_output.content
            else:
                sql_results = agent_output

            return {
                **state,
                "sql_results": str(sql_results),
                "messages": state["messages"]
                + [AIMessage(content="SQL query completed")],
            }
        except Exception as e:
            logging.error(f"SQL Agent error: {e}")
            error_msg = f"SQL query failed: {str(e)}"
            return {
                **state,
                "sql_results": error_msg,
                "error": str(e),
                "messages": state["messages"] + [AIMessage(content=error_msg)],
            }


class VectorAgent:
    def __init__(self, qdrant, embedding):
        self.qdrant = qdrant
        self.embedding = embedding

    def run_vector_search(self, query) -> str:
        """Create vector search tool using LangChain Qdrant integration"""

        vector_store_crd = QdrantVectorStore(
            client=self.qdrant, collection_name="crd_data", embedding=self.embedding
        )

        vector_store_resource = QdrantVectorStore(
            client=self.qdrant,
            collection_name="resource_data",
            embedding=self.embedding,
        )

        vector_store_controller = QdrantVectorStore(
            client=self.qdrant,
            collection_name="controller_data",
            embedding=self.embedding,
        )

        result_crd = vector_store_crd.similarity_search(query, k=2)
        logging.info(f"Fetched {len(result_crd)} document from CRD")

        result_resource = vector_store_resource.similarity_search(query, k=2)
        logging.info(f"Fetched {len(result_resource)} document from resource")

        result_controller = vector_store_controller.similarity_search(query, k=2)
        logging.info(f"Fetched {len(result_controller)} document from controller")

        doc = result_crd + result_resource + result_controller
        results = []
        for d in doc:
            if hasattr(d, "page_content"):
                results.append(d.page_content)

        return "\n\n".join(results)

    def __call__(self, state: AgentState) -> AgentState:
        """Vector agent that uses LangChain RAG chain"""
        query = state.get("query", "")
        if state.get("sql_results") is not None:
            query += "\n\n SQL RESULT: " + str(state["sql_results"])

        try:
            vector_results = self.run_vector_search(query)
            logging.info(f"Vector Agent completed search successfully")

            return {
                **state,
                "vector_results": vector_results,
                "messages": state["messages"]
                + [AIMessage(content="Vector search completed")],
            }
        except Exception as e:
            logging.error(f"Vector Agent error: {e}")
            error_msg = f"Vector search failed: {str(e)}"
            return {
                **state,
                "vector_results": error_msg,
                "error": str(e),
                "messages": state["messages"] + [AIMessage(content=error_msg)],
            }


class SynthesizerAgent:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.prompt = open(prompt_file, "r").read()

    def __call__(self, state: AgentState) -> AgentState:
        """Synthesizer agent that combines results and generates final response"""
        query = state.get("query", "")
        sql_results = state.get("sql_results", "")
        vector_results = state.get("vector_results", "")

        synthesis_prompt = self.prompt.format(
            query=query,
            sql_results={sql_results if sql_results else "No SQL data available"},
            vector_results={
                vector_results if vector_results else "No vector data available"
            },
        )

        try:
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            final_response = response.content

            logging.info("Synthesizer Agent completed successfully")

            return {
                **state,
                "final_response": final_response,
                "messages": state["messages"] + [AIMessage(content=final_response)],
            }
        except Exception as e:
            logging.error(f"Synthesizer Agent error: {e}")
            error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            return {
                **state,
                "final_response": error_response,
                "error": str(e),
                "messages": state["messages"] + [AIMessage(content=error_response)],
            }
