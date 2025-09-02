from io import BytesIO
from fastapi import UploadFile
from fastapi.datastructures import Headers
import pytest
from unittest.mock import AsyncMock

from src.session.file_uploads import FileUpload
from src.directors.report_director import report_on_file_upload

@pytest.mark.asyncio
async def test_report_on_file_upload(mocker):

    file_upload = FileUpload(
        uploadId="1",
        filename="test.txt",
        content="test sustainability document content",
        contentType="text/plain",
        size=4
    )

    mock_handle_file_upload = mocker.patch("src.directors.report_director.handle_file_upload", return_value=file_upload)

    # Mock the ReportAgent
    mock_report_agent_class = mocker.patch("src.directors.report_director.ReportAgent")
    mock_report_agent_instance = AsyncMock()
    mock_report_agent_instance.invoke.return_value = (
        "# ESG Analysis Report\n\nThis is a generated ESG report."
    )
    mock_report_agent_class.return_value = mock_report_agent_instance

    # Mock the config
    mock_config = mocker.patch("src.directors.report_director.config")
    mock_config.report_agent_llm = "mistral"
    mock_config.report_agent_model = "mistral-large-latest"

    headers = Headers({"content-type": "text/plain"})
    file = BytesIO(b"test content")
    request_upload_file = UploadFile(
        file=file, size=12, headers=headers, filename="test.txt"
    )
    response = await report_on_file_upload(request_upload_file)

    mock_handle_file_upload.assert_called_once_with(request_upload_file)
    mock_report_agent_class.assert_called_once_with("mistral", "mistral-large-latest")
    mock_report_agent_instance.invoke.assert_called_once_with(
        "test sustainability document content"
    )
    assert response == {
        "filename": "test.txt",
        "id": "1",
        "report": "# ESG Analysis Report\n\nThis is a generated ESG report."
    }
