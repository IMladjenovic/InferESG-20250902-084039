
from typing import TypedDict
from fastapi import UploadFile

from src.utils.scratchpad import clear_scratchpad, update_scratchpad
from src.utils.file_utils import handle_file_upload
from src.agents.report_agent import ReportAgent
from src.utils import Config

config = Config()

class FileUploadReport(TypedDict):
    id: str
    filename: str | None
    report: str | None

async def report_on_file_upload(upload:UploadFile) -> FileUploadReport:

    file = handle_file_upload(upload)

    update_scratchpad(result=file["content"])

    # Create and invoke the Report Agent to generate ESG report
    report_agent = ReportAgent(config.report_agent_llm, config.report_agent_model)
    content = file["content"] or ""
    report = await report_agent.invoke(content)

    clear_scratchpad()

    return {"filename": file["filename"], "id": file["uploadId"], "report": report}
