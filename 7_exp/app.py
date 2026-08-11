import asyncio
import os
import webbrowser
from google import genai
from google.genai import types
from fastmcp import FastMCP, Client

# 1. Initialize the FastMCP Server and define a basic tool
mcp = FastMCP("GitHub Desktop Utility")

@mcp.tool()
def open_github_repo(repo_name: str) -> str:
    """
    Opens a given GitHub repository or organization URL in the default web browser.
    
    Args:
        repo_name: The name or path of the repository (e.g., 'octocat/Hello-World' or just 'google')
    """
    cleaned_name = repo_name.strip().strip("/")
    if not cleaned_name:
        url = "https://github.com"
    elif "/" in cleaned_name or not "http" in cleaned_name:
        url = f"https://github.com/{cleaned_name}"
    else:
        url = cleaned_name

    try:
        print(f"\n[Tool Execution] Opening browser to: {url}")
        webbrowser.open(url)
        return f"Successfully opened the URL in the browser: {url}"
    except Exception as e:
        return f"Failed to open browser: {str(e)}"


async def main():
    # Verify API key configuration
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Run: export GEMINI_API_KEY='your_key'")
        return

    print("Initializing Gemini Client and MCP Server bindings...")

    # 2. Setup standard Google GenAI client
    gemini_client = genai.Client()

    # 3. Create an in-process MCP client context targeting our FastMCP server instance
    async with Client(mcp) as mcp_client:
        # Fetch available MCP tools dynamically
        mcp_tools = await mcp_client.list_tools()
        
        # Convert MCP tools to Google GenAI compatible declarations
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema,
                    )
                    for tool in mcp_tools.tools
                ]
            )
        ]

        # Agent instruction / prompt loop setup
        prompt = "Can you open the official repository for LangChain on GitHub for me?"
        print(f"\nUser Prompt: {prompt}")

        # 4. First turn: Send user prompt and tool schema to Gemini Flash
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',  # Fast, lightweight model optimized for tool use
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0.0
            )
        )

        # 5. Check if the model decided to call a tool
        if response.function_calls:
            for function_call in response.function_calls:
                print(f"\n[Agentic Loop] Gemini requested tool call: '{function_call.name}'")
                print(f"Arguments passed: {function_call.args}")

                tool_name = function_call.name
                tool_args = function_call.args

                # Execute the tool via MCP client session
                if tool_name == "open_github_repo":
                    tool_result = await mcp_client.call_tool(tool_name, tool_args)
                    # Extract string output from MCP text content block
                    output_message = tool_result.content[0].text if tool_result.content else "Done"
                    print(f"[Tool Result]: {output_message}")

                    # 6. Second turn: Send the tool result back to Gemini so it can formulate a final answer
                    final_response = await gemini_client.aio.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            prompt,
                            response.candidates[0].content, # Model's tool call request
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=tool_name,
                                        response={"result": output_message}
                                    )
                                ]
                            )
                        ],
                        config=types.GenerateContentConfig(tools=gemini_tools)
                    )
                    print(f"\nAI Agent Final Response:\n{final_response.text}")
        else:
            print(f"\nAI Agent Response (No tool needed):\n{response.text}")

if __name__ == "__main__":
    asyncio.run(main())
