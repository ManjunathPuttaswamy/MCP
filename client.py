import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


async def main():
    """_summary_
    """
    client = MultiServerMCPClient({
        "calculator": {
            "command": "python",
            "args": ["calculator.py"],   # make sure this file exists / is on PATH
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",  # make sure your server is running
            "transport": "streamable_http",
        },
    })
    
    tools = await client.get_tools()
    
    model = ChatGroq(model="deepseek-r1-distill-llama-70b")
    
    agent = create_react_agent(model, tools)
    
    response = await agent.ainvoke(
        #{"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
        {"messages": [{"role": "user", "content": "what is a weather in banglore?"}]}
    )

    # response is a dict; the conversation is under "messages"
    msgs = response.get("messages", [])
    if not msgs:
        print("No messages returned:", response)
        return
    
    last = msgs[-1]
    # Works whether it's a LangChain Message object or a plain dict
    content = getattr(last, "content", None) or last.get("content", str(last))
    print(content)

asyncio.run(main())