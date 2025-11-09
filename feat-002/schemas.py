from pydantic import BaseModel, Field

class WebSearchInput(BaseModel):
    """Input for web search queries."""

    query: str = Field(description="Search query string")
    limit: int = Field(default=10, description="Maximum number of results to return")

class Source(BaseModel):
    """Schema for source used by an agent"""

    url: str = Field(description="URL of the source.")


class AgentResponse(BaseModel):
    """Job listing information"""

    answer: str = Field(description="The answer to the question")
    sources: list[Source] = Field(
        default_factory=list, 
        scription="The sources used to answer the question"
    )

