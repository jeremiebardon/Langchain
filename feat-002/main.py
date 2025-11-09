from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_core.messages import ToolMessage

from langchain_tavily import TavilySearch

from prompt import PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import WebSearchInput, AgentResponse


load_dotenv()

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
    
@tool(args_schema=WebSearchInput)
def search_web(query: str, limit: int = 10) -> str:
    """Search the web for information.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    try:
        tavily = TavilySearch(
            max_results=limit,
            topic="general"
        )

        response = tavily.invoke(query)

        return response
    except Exception as e:
        return f"Search failed: {str(e)}"

def main():

    llm = init_chat_model(
        model="gpt-4",
        model_provider="openai", 
        temperature=0.2,
        timeout=30,
    )
    
    agent = create_agent(
        llm, 
        tools=[search_web],
        system_prompt=PROMPT_WITH_FORMAT_INSTRUCTIONS,
        response_format=AgentResponse,
        middleware=[handle_tool_errors]
    )

    result = agent.invoke(
        {
            "messages": [{ "role": "user", "content": "I look for Full stack AI engineer job oriented typescript and dev but with a touch of AI, find at leat 8 relevant jobs" }],
        }
    )

    response = result["structured_response"]
    print(response.answer)

if __name__ == "__main__":
    main()
