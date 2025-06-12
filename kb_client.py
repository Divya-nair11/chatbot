import asyncio
import sys
import os
import shutil
import subprocess
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  
    allow_headers=["*"],
)

class BedrockKBMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.web_session: Optional[ClientSession] = None
        self.gmail_session: Optional[ClientSession] = None  
        self.agent: Optional[AgentExecutor] = None
        self.llm = None
        self.kb_tools = []
        self.web_tools = []
        self.gmail_tools = []  # Added for Gmail tools
        
        self.aws_config = {
            "AWS_ACCESS_KEY_ID": "",
            "AWS_SECRET_ACCESS_KEY": "", 
            "AWS_REGION": "us-east-2",  
            "KNOWLEDGE_BASE_ID": ""
        }

        self.mcp_servers = {  # Added for Gmail
            "gmail": {
                "command": "npx",
                "args": ["@gongrzhe/server-gmail-autoauth-mcp"]
            }
        }

    async def connect_to_kb_server(self):
        env = os.environ.copy()
        env.update(self.aws_config)
        
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-aws-kb-retrieval"],
            env=env
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await self.session.initialize()
            
            kb_tools = await load_mcp_tools(self.session)
            for tool in kb_tools:
                if tool.name == "retrieve_from_aws_kb":
                    tool.args_schema["knowledgeBaseId"] = self.aws_config["KNOWLEDGE_BASE_ID"]
            self.kb_tools.extend(kb_tools)
            
        except Exception as e:
            print(f"Failed to connect to AWS KB server: {str(e)}")
            raise

    async def connect_to_web_search_server(self):
        """Connect to the Tavily MCP server"""
        env = os.environ.copy()
        env["TAVILY_API_KEY"] = "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"
        
        tavily_mcp_path = os.path.abspath("C:/Users/divya.p/Documents/web_search/tavily-mcp/build/index.js")
        
        server_params = StdioServerParameters(
            command="node",
            args=[tavily_mcp_path],
            env=env
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.web_session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await self.web_session.initialize()

            self.web_tools = await load_mcp_tools(self.web_session)
            
        except Exception as e:
            print(f"Failed to connect to Tavily Web Search server: {str(e)}")
            raise

    async def connect_to_gmail_server(self):
        """Connect to the Gmail MCP server"""
        env = os.environ.copy()
        
        # Check if npx is available
        npx_path = shutil.which("npx")
        if not npx_path:
            raise FileNotFoundError("npx not found. Ensure Node.js and npm are installed and added to PATH.")

        # Load OAuth credentials from gcp-oauth-keys.json
        oauth_keys_path = os.path.abspath("C:/Users/divya.p/Documents/chatbot/gcp-oauth.keys.json")
        if not os.path.exists(oauth_keys_path):
            raise FileNotFoundError("gcp-oauth-keys.json not found. Ensure the file exists in the project directory.")
        
        # Create the .gmail-mcp directory if it doesn't exist
        gmail_mcp_dir = os.path.join(os.path.expanduser("~"), ".gmail-mcp")
        if not os.path.exists(gmail_mcp_dir):
            try:
                os.makedirs(gmail_mcp_dir, exist_ok=True)
                print(f"Created directory: {gmail_mcp_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to create .gmail-mcp directory: {str(e)}")

        # Copy gcp-oauth-keys.json to .gmail-mcp as gcp-oauth.keys.json (expected by the server)
        expected_oauth_path = os.path.join(gmail_mcp_dir, "gcp-oauth.keys.json")
        if not os.path.exists(expected_oauth_path):
            try:
                shutil.copyfile(oauth_keys_path, expected_oauth_path)
                print(f"Copied OAuth keys to: {expected_oauth_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to copy OAuth keys to .gmail-mcp: {str(e)}")

        # Check if credentials.json exists; if not, trigger authentication
        credentials_path = os.path.join(gmail_mcp_dir, "credentials.json")
        if not os.path.exists(credentials_path):
            print("Gmail credentials not found. Initiating authentication process...")
            try:
                # Run the authentication command
                auth_env = os.environ.copy()
                auth_env["GMAIL_OAUTH_PATH"] = expected_oauth_path
                result = subprocess.run(
                    [npx_path, "@gongrzhe/server-gmail-autoauth-mcp", "auth"],
                    env=auth_env, capture_output=True, text=True, check=True
                )
                print("Authentication command output:", result.stdout)
                print("Authentication command error (if any):", result.stderr)
                if "Authentication completed successfully" in result.stdout:
                    print("Authentication successful. Proceeding with Gmail MCP server connection...")
                else:
                    raise RuntimeError("Authentication failed. Please check the output above for details.")
            except subprocess.CalledProcessError as cpe:
                print("Authentication command failed with exit code:", cpe.returncode)
                print("Authentication command output:", cpe.stdout)
                print("Authentication command error:", cpe.stderr)
                raise RuntimeError("Failed to authenticate Gmail MCP server.")
            except Exception as sub_e:
                print("Error running authentication command:", str(sub_e))
                raise

        # Set the environment variable for the Gmail MCP server
        env["GMAIL_OAUTH_PATH"] = expected_oauth_path
        
        server_params = StdioServerParameters(
            command=npx_path,
            args=self.mcp_servers["gmail"]["args"],
            env=env
        )
        
        try:
            print("Attempting to start Gmail MCP server...")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            print("Gmail MCP server started successfully.")
            self.gmail_session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
            await self.gmail_session.initialize()
            print("Gmail session initialized.")

            self.gmail_tools = await load_mcp_tools(self.gmail_session)
            for tool in self.gmail_tools:
                print(f"Loaded Gmail tool: {tool.name}")

        except Exception as e:
            print(f"Failed to connect to Gmail MCP server: {str(e)}")
            try:
                result = subprocess.run(
                    [npx_path] + self.mcp_servers["gmail"]["args"],
                    env=env, capture_output=True, text=True, check=True
                )
                print("Manual command output:", result.stdout)
                print("Manual command error:", result.stderr)
            except subprocess.CalledProcessError as cpe:
                print("Manual command failed with exit code:", cpe.returncode)
                print("Manual command output:", cpe.stdout)
                print("Manual command error:", cpe.stderr)
            except Exception as sub_e:
                print("Error running manual command:", str(sub_e))
            raise

    def _setup_bedrock_agent(self):
        self.llm = ChatBedrock(
            provider="anthropic",
            model_id="arn:aws:bedrock:us-east-2:528757829695:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            temperature=0,
            max_tokens=3000,
            region_name=self.aws_config["AWS_REGION"],
            aws_access_key_id=self.aws_config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=self.aws_config["AWS_SECRET_ACCESS_KEY"],
        )

        # First, create an agent with KB and Gmail tools
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to a company knowledge base and Gmail.

    - Use `retrieve_from_aws_kb` to fetch information from the KB.
    - If the KB provides a relevant answer, return it immediately with [KNOWLEDGE BASE] label.
    - If the query involves sending an email, use `send_email` to send an email via Gmail. Label the response with [GMAIL].
    - If the KB does not provide a relevant answer and the query is not email-related, return "KB_NOT_FOUND" to indicate failure.
    """),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Combine KB and Gmail tools for the first agent
        all_tools = self.kb_tools + self.gmail_tools
        kb_agent = create_tool_calling_agent(self.llm, all_tools, prompt)
        self.kb_agent = AgentExecutor(agent=kb_agent, tools=all_tools, verbose=True, max_iterations=2)

        # Then, create a separate agent for Tavily web search (to be used only if KB fails)
        web_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with access to web search via Tavily.

    - Use `tavily_web_search` to fetch information from the web.
    - If the search provides a relevant answer, return it immediately with [WEB SEARCH] label.
    - If the web search does not provide a relevant answer, return "WEB_NOT_FOUND" to indicate failure.
    """),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        web_agent = create_tool_calling_agent(self.llm, self.web_tools, web_prompt)
        self.web_agent = AgentExecutor(agent=web_agent, tools=self.web_tools, verbose=True, max_iterations=15)

    async def process_query(self, query: str) -> str:
        if not self.kb_agent or not self.session:
            raise RuntimeError("Agent not initialized. Call connect methods first.")

        try:
            # Try KB retrieval or Gmail operations
            kb_result = await self.kb_agent.ainvoke({"input": query})
            kb_output = kb_result.get("output", "No response generated")

            if isinstance(kb_output, list):
                cleaned_outputs = []
                for item in kb_output:
                    if isinstance(item, dict):
                        cleaned_outputs.append(item.get("text", ""))
                    else:
                        cleaned_outputs.append(str(item))
                kb_output = " ".join(cleaned_outputs)
            elif isinstance(kb_output, dict):
                kb_output = kb_output.get("text", "")

            if kb_output != "KB_NOT_FOUND" and ("[KNOWLEDGE BASE]" in kb_output or "[GMAIL]" in kb_output):
                return kb_output.replace("[KNOWLEDGE BASE]", "").replace("[GMAIL]", "").strip()

            # If KB fails, try Tavily web search
            if self.web_agent and self.web_session:
                web_result = await self.web_agent.ainvoke({"input": query})
                web_output = web_result.get("output", "No response generated")
                
                if isinstance(web_output, list):
                # Process list items - look for the first valid response
                    for item in web_output:
                        if isinstance(item, dict):
                            item_text = item.get("text", "")
                            if "[WEB SEARCH]" in item_text:
                                return item_text.replace("[WEB SEARCH]", "").strip()
                        elif isinstance(item, str) and "[WEB SEARCH]" in item:
                            return item.replace("[WEB SEARCH]", "").strip()
                    web_output = " ".join(str(item) for item in web_output)
                
                # Check if web search was successful
                if web_output != "WEB_NOT_FOUND" and "[WEB SEARCH]" in web_output:
                    # Extract the actual web search content
                    web_content = web_output.replace("[WEB SEARCH]", "").strip()
                    if web_content:  # Only return if we have actual content
                        return web_content

            # Fallback to LLM if both KB and web search fail
            llm_response = await self.llm.ainvoke([
                SystemMessage(content="Answer using general knowledge. KB and web search failed."),
                HumanMessage(content=query)
            ])
            return f"{llm_response.content}"

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            try:
                llm_response = await self.llm.ainvoke([
                    SystemMessage(content="Answer using general knowledge. KB and web search failed."),
                    HumanMessage(content=query)
                ])
                return f"Answer: [Fallback Response] {llm_response.content}"
            except Exception as fallback_error:
                return f"Error: Unable to process query. {str(fallback_error)}"

    async def cleanup(self):
        try:
            if self.session:
                await self.exit_stack.aclose()
        except Exception as e:
            print(f"Cleanup warning: {str(e)}")
        finally:
            await asyncio.sleep(0.1)

# FastAPI setup
client = BedrockKBMCPClient()

# Pydantic model for request validation
class ChatRequest(BaseModel):
    query: str

# Startup event to initialize the client
@app.on_event("startup")
async def startup_event():
    print("Connecting to MCP servers...")
    await client.connect_to_kb_server()
    await client.connect_to_web_search_server()
    await client.connect_to_gmail_server()  # Added Gmail server connection
    client._setup_bedrock_agent()

# Shutdown event to clean up
@app.on_event("shutdown")
async def shutdown_event():
    await client.cleanup()

# API endpoint to handle chat queries
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await client.process_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)