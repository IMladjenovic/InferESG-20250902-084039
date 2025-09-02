import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.report_agent import ReportAgent


@pytest.mark.asyncio
async def test_report_agent_invoke(mocker):
    """Test that ReportAgent correctly invokes the LLM with proper prompts"""
    
    # Mock the LLM
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "# ESG Report\n\nThis is a test ESG report analyzing the document."
    
    # Mock the PromptEngine
    mock_engine = mocker.patch("src.agents.report_agent.engine")
    mock_engine.load_prompt.return_value = "System prompt for ESG analysis"
    
    # Create ReportAgent instance
    agent = ReportAgent("mistral", "mistral-large-latest")
    agent.llm = mock_llm
    
    # Test document content
    document_content = "This is a test sustainability document from XYZ Corp."
    
    # Invoke the agent
    result = await agent.invoke(document_content)
    
    # Verify the prompt template was loaded
    mock_engine.load_prompt.assert_called_once_with("esg-report-system")
    
    # Verify the LLM was called with correct parameters
    mock_llm.chat.assert_called_once_with(
        "mistral-large-latest",
        "System prompt for ESG analysis",
        f"Create a report to flag greenwashing\n\nDocument content:\n{document_content}"
    )
    
    # Verify the result
    assert result == "# ESG Report\n\nThis is a test ESG report analyzing the document."


@pytest.mark.asyncio
async def test_report_agent_with_empty_document(mocker):
    """Test that ReportAgent handles empty document content gracefully"""
    
    # Mock the LLM
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "# ESG Report\n\nNo content to analyze."
    
    # Mock the PromptEngine
    mock_engine = mocker.patch("src.agents.report_agent.engine")
    mock_engine.load_prompt.return_value = "System prompt for ESG analysis"
    
    # Create ReportAgent instance
    agent = ReportAgent("mistral", "mistral-large-latest")
    agent.llm = mock_llm
    
    # Test with empty document content
    document_content = ""
    
    # Invoke the agent
    result = await agent.invoke(document_content)
    
    # Verify the LLM was called
    mock_llm.chat.assert_called_once()
    
    # Verify the result
    assert result == "# ESG Report\n\nNo content to analyze."


@pytest.mark.asyncio
async def test_report_agent_llm_error_handling(mocker):
    """Test that ReportAgent properly handles LLM errors"""
    
    # Mock the LLM to raise an exception
    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = Exception("LLM service unavailable")
    
    # Mock the PromptEngine
    mock_engine = mocker.patch("src.agents.report_agent.engine")
    mock_engine.load_prompt.return_value = "System prompt for ESG analysis"
    
    # Create ReportAgent instance
    agent = ReportAgent("mistral", "mistral-large-latest")
    agent.llm = mock_llm
    
    # Test document content
    document_content = "Test document"
    
    # Verify that the exception is propagated
    with pytest.raises(Exception, match="LLM service unavailable"):
        await agent.invoke(document_content)


def test_report_agent_decorator():
    """Test that the ReportAgent class has correct metadata from the decorator"""
    
    assert ReportAgent.name == "ReportAgent"
    assert "ESG reports" in ReportAgent.description
    assert "sustainability documents" in ReportAgent.description
    assert ReportAgent.tools == []