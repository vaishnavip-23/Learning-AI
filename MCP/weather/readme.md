MCP Server 
We’ll build a server that exposes two tools: get_alerts and get_forecast. Then we’ll connect the server to an MCP host (in this case, Claude for Desktop).

MCP servers can provide three main types of capabilities:
Resources: File-like data that can be read by clients (like API responses or file contents)
Tools: Functions that can be called by the LLM (with user approval)
Prompts: Pre-written templates that help users accomplish specific tasks