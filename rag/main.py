import os

from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma

from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_core.messages import ToolMessage
from langchain_core.documents import Document
from langchain_core.runnables import chain

from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from prompt import PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")

vector_store = Chroma(
    collection_name="nike_sec_report",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def embedd_document(doc: Document) -> List[float]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "nike.pdf")

    loader = PyPDFLoader(file_path)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)

@chain
def retriever() -> List[Document]:
    # Search type could be MMR (Maximal Marginal Relevance) to include more diverse results, useful when searching a subject in a documentation for example
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retriever

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

def main():
    # doc, score = results[0]
    # print(f"Score: {score}\nContent: {doc.page_content}")
    res = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )

    print(res[0])

    """ llm = init_chat_model(
        model="gpt-4",
        model_provider="openai", 
        temperature=0.2,
        timeout=30,
    )
    
    agent = create_agent(
        llm, 
        tools=[],
        system_prompt="You should answer question about nike security exchange report based on the documents provided. You should not invent any information that is not present in the documents.",
        middleware=[handle_tool_errors]
    )

    result = agent.invoke(
        {
            "messages": [{ "role": "user", "content": "What could you tell me about nike culture ?" }],
        }
    )

    response = result["structured_response"]
    print(response.answer) """

if __name__ == "__main__":
    main()
