from src.agents import Agent, agent
from src.prompts import PromptEngine

engine = PromptEngine()


@agent(
    name="ReportAgent",
    description="This agent generates ESG reports about sustainability documents uploaded by users, analyzing environmental, social, and governance aspects to identify potential greenwashing",
    tools=[],
)
class ReportAgent(Agent):
    async def invoke(self, document_content: str) -> str:
        """
        Generate an ESG report analyzing the provided document content.
        
        Args:
            document_content: The content of the sustainability document to analyze
            
        Returns:
            A markdown-formatted ESG report analyzing the document
        """
        # Load the ESG report system prompt template
        system_prompt = engine.load_prompt("esg-report-system")
        
        # User prompt to create the report
        user_prompt = "Create a report to flag greenwashing"
        
        # Generate the ESG report using the LLM
        report = await self.llm.chat(
            self.model, 
            system_prompt, 
            f"{user_prompt}\n\nDocument content:\n{document_content}"
        )
        
        return report