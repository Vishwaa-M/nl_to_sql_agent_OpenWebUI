import pytest
import pytest_asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

# Set the project root path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tools directly for unit tests
from tools.sql_tools import get_relevant_schema, get_few_shot_examples, execute_sql_query
from tools.analysis_tools import summarize_results, plan_visualizations, generate_visualization_code
from psycopg import Error as PsycopgError
import sqlvalidator  # Added for SQL validation tests
from core.agent import get_agent_executor  # Updated import for agent executor

# --- Mock Fixtures for Unit Tests ---

@pytest.fixture
def mock_vector_store():
    with patch('tools.sql_tools.get_vector_store_manager', new_callable=MagicMock) as m1, \
         patch('tools.memory_management.get_vector_store_manager', new_callable=MagicMock) as m2:
        mock_store = MagicMock()
        mock_store.similarity_search = AsyncMock()
        mock_store.add_documents = AsyncMock()
        m1.return_value = mock_store
        m2.return_value = mock_store
        yield mock_store

@pytest.fixture
def mock_llm_client():
    with patch('tools.analysis_tools.get_llm_client') as mock_get:
        mock_client = MagicMock()
        mock_client.invoke_chat_completion = AsyncMock()
        mock_get.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_db_connection():
    with patch('tools.sql_tools.get_db_connection') as mock_get:
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__aenter__.return_value = mock_cursor
        mock_get.return_value.__aenter__.return_value = mock_conn
        yield mock_cursor

@pytest.fixture
def mock_sqlvalidator():
    with patch('tools.sql_tools.sqlvalidator') as mock_sqlv:
        yield mock_sqlv

# --- Unit Tests ---

@pytest.mark.asyncio
async def test_get_relevant_schema_success(mock_vector_store):
    mock_vector_store.similarity_search.return_value = [{'text': 'Table: users'}]
    result = await get_relevant_schema.ainvoke({"question": "find user orders"})
    assert "Table: users" in result

@pytest.mark.asyncio
async def test_get_few_shot_examples_success(mock_vector_store):
    mock_vector_store.similarity_search.return_value = [{'text': 'SQL Query: SELECT count(*)'}]
    result = await get_few_shot_examples.ainvoke({"question": "how many users"})
    assert "SELECT count(*)" in result

@pytest.mark.asyncio
async def test_execute_sql_query_success(mock_db_connection, mock_sqlvalidator):
    # Mock sqlvalidator to return True (valid query)
    mock_sqlvalidator.parse.return_value = True
    # Mock database execution
    mock_db_connection.fetchall.return_value = [("Alice",)]
    mock_db_connection.description = [("name",)]
    result = await execute_sql_query.ainvoke({"sql_query": "SELECT name FROM users;"})
    assert result["sql_error"] is None
    assert result["query_result"] == [{'name': 'Alice'}]
    mock_sqlvalidator.parse.assert_called_once_with("SELECT name FROM users;")

@pytest.mark.asyncio
async def test_execute_sql_query_disallows_non_select(mock_sqlvalidator):
    # No need to mock sqlvalidator.parse since non-SELECT check happens first
    result = await execute_sql_query.ainvoke({"sql_query": "UPDATE users SET name = 'Eve';"})
    assert "Security Error" in result["sql_error"]
    mock_sqlvalidator.parse.assert_not_called()

@pytest.mark.asyncio
async def test_execute_sql_query_invalid_syntax(mock_sqlvalidator):
    # Mock sqlvalidator to return False (invalid query)
    mock_sqlvalidator.parse.return_value = False
    # Use a valid SELECT query with invalid syntax
    result = await execute_sql_query.ainvoke({"sql_query": "SELECT * FROM users WHERE name = 'Alice;"})
    assert "SQL Validation Error: Invalid SQL syntax" in result["sql_error"]
    mock_sqlvalidator.parse.assert_called_once_with("SELECT * FROM users WHERE name = 'Alice;")

@pytest.mark.asyncio
async def test_execute_sql_query_handles_db_error(mock_db_connection, mock_sqlvalidator):
    # Mock sqlvalidator to return True (valid query)
    mock_sqlvalidator.parse.return_value = True
    # Mock database execution to raise an error
    mock_db_connection.execute.side_effect = PsycopgError("relation 'non_existent_table' does not exist")
    result = await execute_sql_query.ainvoke({"sql_query": "SELECT * FROM non_existent_table;"})
    assert "Database Execution Error" in result["sql_error"]
    mock_sqlvalidator.parse.assert_called_once_with("SELECT * FROM non_existent_table;")

@pytest.mark.asyncio
async def test_summarize_results_success(mock_llm_client):
    mock_llm_client.invoke_chat_completion.return_value = "The total sales were $150."
    result = await summarize_results.ainvoke({"question": "Total sales?", "query_result": [{"total": 150}]})
    assert result == "The total sales were $150."

@pytest.mark.asyncio
async def test_plan_visualizations_success(mock_llm_client):
    mock_llm_client.invoke_chat_completion.return_value = '{"charts": [{"chart_type": "bar", "title": "Sales", "x_axis": "p", "y_axis": "s", "explanation": "..."}]}'
    result = await plan_visualizations.ainvoke({"question": "Show sales", "data": [{"p": "A", "s": 100}]})
    assert "error" not in result
    assert result["plan"]["charts"][0]["chart_type"] == "bar"

@pytest.mark.asyncio
async def test_plan_visualizations_handles_invalid_json(mock_llm_client):
    mock_llm_client.invoke_chat_completion.return_value = "{'charts':"
    result = await plan_visualizations.ainvoke({"question": "Show sales", "data": [{"p": "A", "s": 100}]})
    assert "error" in result

@pytest.mark.asyncio
async def test_generate_visualization_code_success(mock_llm_client):
    mock_llm_client.invoke_chat_completion.return_value = "fig = px.bar(df, x='p', y='s')"
    chart_plan = {"chart_type": "bar", "x_axis": "p", "y_axis": "s", "title": "Sales"}
    result = await generate_visualization_code.ainvoke({"question": "Chart sales", "chart_plan": chart_plan})
    assert "error" not in result
    assert result["code"] == "fig = px.bar(df, x='p', y='s')"

# --- Integration Test ---

@pytest.mark.asyncio
async def test_full_agent_flow():
    """
    Tests the main agent flow. Mocks are applied BEFORE the agent is compiled
    to solve the checkpointer patch timing issue.
    """
    # Mock checkpointer with proper AsyncPostgresSaver interface
    mock_checkpointer_instance = MagicMock(spec=['aget', 'alist', 'aput', 'get_next_version', 'setup', 'aput_writes', 'aget_tuple'])
    mock_checkpointer_instance.aget = AsyncMock(return_value=None)
    mock_checkpointer_instance.alist = AsyncMock(return_value=[])
    mock_checkpointer_instance.aput = AsyncMock(return_value=None)
    mock_checkpointer_instance.get_next_version = AsyncMock(return_value=None)
    mock_checkpointer_instance.setup = AsyncMock(return_value=None)
    mock_checkpointer_instance.aput_writes = AsyncMock(return_value=None)
    mock_checkpointer_instance.aget_tuple = AsyncMock(return_value=None)  # Changed to plain None to avoid coroutine
    
    with patch('core.agent.AsyncPostgresSaver') as mock_checkpointer_class, \
         patch('agents.nodes.get_llm_client') as mock_llm_class, \
         patch('tools.sql_tools.get_db_connection') as mock_db, \
         patch('tools.sql_tools.get_vector_store_manager') as mock_vs, \
         patch('tools.analysis_tools.get_llm_client') as mock_analysis_llm, \
         patch('tools.memory_management.get_vector_store_manager') as mock_mem_vs, \
         patch('core.sandbox.safe_execute_visualization_code') as mock_sandbox:
    
        # Configure mocks
        mock_checkpointer_class.from_conn_string.return_value = mock_checkpointer_instance
    
        llm_client = AsyncMock()
        llm_client.invoke_chat_completion.side_effect = [
            '{"route": "sql_query"}', 'SELECT * FROM sales', 'The summary is sales were high.',
            '{"charts": [{"chart_type": "bar", "title": "Sales", "x_axis": "p", "y_axis": "s", "explanation": "..."}]}',
            "fig = px.bar(df, x='p', y='s')", "<html><body><h1>Dashboard</h1></body></html>", '{"facts_to_save": ["User is interested in sales data."]}'
        ]
        mock_llm_class.return_value = llm_client
        mock_analysis_llm.return_value = llm_client
    
        vector_store_client = AsyncMock(similarity_search=AsyncMock(return_value=[{'text': 'mocked context'}]), add_documents=AsyncMock())
        mock_vs.return_value = vector_store_client
        mock_mem_vs.return_value = vector_store_client
        
        mock_cursor = AsyncMock(fetchall=AsyncMock(return_value=[('Product A', 100)]), description=[('product',), ('sales',)])
        mock_db.return_value.__aenter__.return_value.cursor.return_value = mock_cursor
        
        mock_sandbox.return_value = {"image_bytes": b"fake_image_bytes", "error": None}
    
        # Mock sqlvalidator for the integration test
        sqlvalidator.parse = MagicMock(return_value=True)  # Match actual sqlvalidator behavior
    
        # Create the agent executor with the mocked checkpointer
        agent_executor = get_agent_executor(checkpointer=mock_checkpointer_instance)
    
        # Arrange
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        agent_input = {"question": "What are sales?", "chat_history": [], "user_id": "test_user_123"}
        
        # Act
        final_state_data = await agent_executor.ainvoke(agent_input, config=config)
    
        # Assert
        assert final_state_data is not None
        assert final_state_data.get("generated_sql") == 'SELECT * FROM sales'
        assert final_state_data.get("summary") == "The summary is sales were high."
        assert final_state_data.get("facts_to_save") == ["User is interested in sales data."]
        assert final_state_data.get("dashboard_html") == "<html><body><h1>Dashboard</h1></body></html>"