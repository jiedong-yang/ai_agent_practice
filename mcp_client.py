import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from typing import List, Any

load_dotenv()

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent('google-gla:gemini-2.5-flash', mcp_servers=[server],
              system_prompt="you are an expert")

async def main() -> None:
    history: List[Any] = list()
    async with agent.run_mcp_servers():
        while True:
            user_prompt = input("Enter your message: ")
            result = await agent.run(user_prompt, message_history=history)
            history = list(result.all_messages())
            print(result.output)

if __name__ == '__main__':
    asyncio.run(main())