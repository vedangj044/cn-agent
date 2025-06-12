from langchain_community.agent_toolkits import SQLDatabaseToolkit

def get_sql_tool(sql_db, llm):
    """Create SQL database tool using LangChain SQLDatabaseToolkit"""

    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
    sql_tool = toolkit.get_tools()[0]

    # Add schema context to the tool description
    sql_tool.description = f"""
    Use this tool to query the SQLite database containing Kubernetes resources.
    
    Available tables:
    - crd_table (crd, last_updated_timestamp, controller_name, names): Custom Resource Definitions
    - controller_table (controller, last_updated_timestamp): Controller information  
    - instance_table (resource_name, crd): Resource instances linked to CRDs
    
    {sql_tool.description}
    """
    return sql_tool