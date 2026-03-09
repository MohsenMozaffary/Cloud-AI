import json
import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import messages_from_dict
from langchain_core.tools import tool

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0,
                 api_key = api_key)

@tool
def retriever_tool(query:str) -> str:
    """
    Retrieve relevant documents from a PDF database.
    """

    return ""

tools = [retriever_tool]
llm = llm.bind_tools(tools)

def lambda_handler(event, context):

    try:
        body = json.loads(event["body"])

        prompt: List[Dict[str, Any]] = body["prompt"]
        
        messages = messages_from_dict(prompt)

        response = llm.invoke(messages)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "result": response.model_dump()
            })
        }
    except Exception as e:

        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }