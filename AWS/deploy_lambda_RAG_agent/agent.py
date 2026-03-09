from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import messages_to_dict
from langchain_aws import AmazonKnowledgeBasesRetriever
import boto3
import json

import requests
import os

bedrock_client = boto3.client("bedrock-agent-runtime")

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id= "KB ID",
    client=bedrock_client,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
)

url_openai = "link_to_openAI_lambda_bounded_to_tools"

def call_openai(payload):

    try:
        response = requests.post(
            url_openai,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if "body" in data:
            data = json.loads(data["body"])

        return data

    except requests.exceptions.Timeout:
        print("Request timed out")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    return None


@tool
def retriever_tool(query: str) -> str:
    """
    Retrieve relevant information from the Bedrock Knowledge Base.

    This tool searches the vector database created from the PDF document
    and returns the most relevant text passages for the given query.

    The tool limits the returned context to avoid exceeding the LLM
    context window.

    Args:
        query: The user's question or search query.

    Returns:
        A string containing the most relevant document excerpts that
        may help answer the question.
    """

    docs = retriever.invoke(query) or []

    print("Total docs returned:", len(docs))

    results = []

    for i, doc in enumerate(docs[:3]):  
        chunk = doc.page_content[:1500] 
        results.append(f"Document {i+1}:\n{chunk}")

    output = "\n\n".join(results)

    print("Final output length:", len(output))

    return output

tools = [retriever_tool]
tools_dict = {our_tool.name: our_tool for our_tool in tools} 


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state:AgentState)->AgentState:
    """Checks if the last message contains tool calls"""

    result = state["messages"][-1]

    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages

    response = call_openai({
    "prompt": messages_to_dict(messages)
    })

    if not response or "result" not in response:
        raise ValueError(f"Invalid API response: {response}")

    message = AIMessage(**response["result"])

    return {'messages': [message]}

tools_dict = {our_tool.name: our_tool for our_tool in tools} 

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}



graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def handler(event, context):

    user_input = event["input"]

    messages = [HumanMessage(content=user_input)]

    result = rag_agent.invoke({"messages": messages})

    return {
        "output": result['messages'][-1].content
    }

