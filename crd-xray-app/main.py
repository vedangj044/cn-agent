# def main():
#     app = create_multi_db_workflow()
#     user_query = "do the logs confirm that the unsealing was successful?"

#     initial_state = AgentState(
#         messages=[HumanMessage(content=user_query)],
#         query=user_query,
#         sql_results=None,
#         vector_results=None,
#         route_decision=None,
#         final_response=None,
#         error=None,
#     )

#     result = app.invoke(initial_state)

#     print("Final Response:", result["final_response"])
#     return result