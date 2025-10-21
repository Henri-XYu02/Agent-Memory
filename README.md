# Agent-Memory

A containerized, user-isolated AI agent system implementing a ReAct (Reasoning and Acting) architecture with automatic container lifecycle management and LRU eviction.

## Overview

Agent-Memory provides a multi-user AI agent platform where each user gets their own isolated Docker container running a ReAct agent. The system handles user requests through a central dispatcher that manages container lifecycle, implements heartbeat monitoring, and automatically cleans up inactive containers.

### Key Features

- **User-Level Isolation**: Each user gets a dedicated Docker container
- **Automatic Container Management**: Lazy container creation with LRU eviction (max 5 concurrent containers)
- **Heartbeat Monitoring**: Background health checks every 60 seconds
- **Automatic Cleanup**: Removes containers inactive for 6+ hours
- **Multi-Provider LLM Support**: AWS Bedrock (Claude, Llama, Titan) and OpenAI
- **ReAct Agent Tools**:
  - Calculator (safe mathematical expressions)
  - Web Search with content extraction (DuckDuckGo + BeautifulSoup)
  - File reading (with safe path validation)
  - Reddit search (via JSON API)
- **Conversation Memory**: Thread-based conversation history using LangGraph checkpointing

## Architecture

```
┌─────────────────┐
│  Client/User    │
└────────┬────────┘
         │ POST /agent/run
         │ {user_id, message}
         │
┌────────▼──────────────────┐
│  Dispatcher Container     │
│  (Flask, Port 5000)       │
│  - Route requests         │
│  - Manage containers      │
│  - Heartbeat monitor      │
└────────┬──────────────────┘
         │
┌────────▼──────────────────┐
│  Redis Container          │
│  (Port 6379)              │
│  - Container metadata     │
│  - User mappings          │
│  - LRU timestamps         │
└───────────────────────────┘
         │
┌────────▼──────────────────────────────────┐
│     Docker Network: agent-network         │
│                                           │
│  ┌─────────────┐      ┌─────────────┐   │
│  │  Agent      │      │  Agent      │   │
│  │  Container  │ ...  │  Container  │   │
│  │  (user_1)   │      │  (user_N)   │   │
│  │  Port 5001  │      │  Port 5001  │   │
│  │             │      │             │   │
│  │ - ReAct     │      │ - ReAct     │   │
│  │ - LangGraph │      │ - LangGraph │   │
│  │ - Tools     │      │ - Tools     │   │
│  └─────────────┘      └─────────────┘   │
│                                           │
│  Max 5 containers (LRU eviction)         │
└───────────────────────────────────────────┘
```

## How It Works

### Request Flow

1. **Client sends request** with `user_id` and `message` to dispatcher
2. **Dispatcher checks Redis** for existing container for that user
3. **Container lifecycle**:
   - If container exists: Route request to it
   - If container doesn't exist:
     - Check if at max capacity (5 containers)
     - If at max: Evict least recently used container
     - Create new container for the user
4. **Agent processes request** using ReAct loop with available tools
5. **Response returned** to client via dispatcher
6. **Timestamp updated** in Redis for LRU tracking

### Container Management

- **Max Containers**: 5 concurrent user containers (configurable)
- **LRU Eviction**: Automatically removes least recently used container when limit reached
- **Heartbeat Monitoring**: Checks container health every 60 seconds
- **Inactivity Cleanup**: Removes containers unused for 6+ hours (runs every 30 minutes)

## Project Structure

```
Agent-Memory/
├── app/
│   ├── agents/
│   │   └── langgraph_react_agent.py    # ReAct agent implementation
│   ├── api/
│   │   ├── dispatcher.py                # Central dispatcher service
│   │   └── agent_service.py             # Individual agent Flask service
│   ├── container/
│   │   ├── docker_manager.py            # Docker container orchestration
│   │   └── redis_manager.py             # Redis metadata management
│   ├── llm/
│   │   └── bedrock_llm.py               # AWS Bedrock LLM wrapper
│   └── tools/
│       ├── calculator.py                # Math expression evaluator
│       ├── web_search_content.py        # Web search + content fetching
│       ├── file_reader.py               # Local file reader
│       └── reddit_search.py             # Reddit search tool
├── data/                                 # Test datasets (GSM8K, HotpotQA)
├── Dockerfile                            # Agent container image
├── Dockerfile.dispatcher                 # Dispatcher container image
├── docker-compose.yml                    # Multi-container orchestration
├── requirements.txt                      # Python dependencies
└── test_agent.ipynb                     # Testing notebook
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11 (for local development)
- AWS credentials (for Bedrock) or OpenAI API key
- Redis (included in docker-compose)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Agent-Memory.git
cd Agent-Memory
```

### 2. Set up environment variables

Create a `.env` file or set environment variables:

```bash
# LLM Provider Configuration
AGENT_PROVIDER=bedrock          # or 'openai'
AGENT_MODEL=claude-3-sonnet     # or other supported models

# AWS Bedrock (if using)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token    # if using temporary credentials

# OpenAI (if using)
OPENAI_API_KEY=your_openai_key

# Container Management
MAX_CONTAINERS=5
INACTIVE_HOURS=6
REDIS_HOST=redis
REDIS_PORT=6379
```

### 3. Build the Docker images

```bash
# Build agent container image
docker build -t agent-memory:latest -f Dockerfile .

# Build dispatcher image (done automatically by docker-compose)
docker-compose build
```

### 4. Start the services

```bash
docker-compose up -d
```

This will start:
- Redis container (port 6379)
- Dispatcher container (port 5000)

Agent containers will be created dynamically as users make requests.

## Usage

### Send a request to the agent

```bash
curl -X POST http://localhost:5000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "What is the capital of France?",
    "session_id": "default"
  }'
```

### API Endpoints

#### Dispatcher API (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/run` | POST | Run agent for a user |
| `/health` | GET | Check dispatcher health |
| `/containers` | GET | List all active containers |
| `/containers/<user_id>` | GET | Get specific user's container |
| `/containers/<user_id>` | DELETE | Stop user's container |
| `/cleanup` | POST | Manually trigger cleanup |

#### Request Format

```json
{
  "user_id": "unique_user_id",
  "message": "Your question or task",
  "session_id": "optional_session_id",
  "system_prompt": "optional_custom_system_prompt"
}
```

#### Response Format

```json
{
  "user_id": "user123",
  "session_id": "default",
  "response": "The capital of France is Paris.",
  "message_count": 2
}
```

### Example Usage with Python

```python
import requests

response = requests.post(
    "http://localhost:5000/agent/run",
    json={
        "user_id": "user123",
        "message": "Search for recent AI news and summarize the top 3 results",
        "session_id": "news_session"
    }
)

result = response.json()
print(result["response"])
```

## Available Agent Tools

### 1. Calculator
Evaluates mathematical expressions safely.

```python
# Example: "What is 15 * 7 + 23?"
# Agent will use calculator("15 * 7 + 23")
```

### 2. Web Search with Content
Searches the web and extracts content from results.

```python
# Example: "Search for Python asyncio tutorial"
# Agent will use search_web_with_content("Python asyncio tutorial", max_results=3)
```

### 3. Fetch Web Page
Directly fetches and extracts content from a URL.

```python
# Example: "What's on the homepage of example.com?"
# Agent will use fetch_web_page("https://example.com")
```

### 4. File Reader
Reads local files safely.

```python
# Example: "Read the contents of data.txt"
# Agent will use read_file("data.txt")
```

### 5. Reddit Search
Searches Reddit posts.

```python
# Example: "Find recent posts about AI on Reddit"
# Agent will use reddit_search("AI", max_results=5)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONTAINERS` | 5 | Maximum concurrent user containers |
| `INACTIVE_HOURS` | 6 | Hours before inactive container cleanup |
| `REDIS_HOST` | redis | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `AGENT_IMAGE` | agent-memory:latest | Docker image for agent containers |
| `DISPATCHER_PORT` | 5000 | Dispatcher service port |
| `AGENT_PROVIDER` | bedrock | LLM provider (bedrock or openai) |
| `AGENT_MODEL` | claude-3-sonnet | Model name |

### Supported LLM Models

#### AWS Bedrock
- Claude 3.5 Sonnet (default)
- Claude 3 Opus
- Claude 3 Haiku
- Llama 2/3 models
- Titan models

#### OpenAI
- GPT-4
- GPT-3.5-turbo
- Other OpenAI chat models

## Monitoring

### Check active containers

```bash
curl http://localhost:5000/containers
```

### Check specific user's container

```bash
curl http://localhost:5000/containers/user123
```

### View logs

```bash
# Dispatcher logs
docker logs agent-dispatcher

# Specific agent container
docker logs agent-user123
```

### Manually trigger cleanup

```bash
curl -X POST http://localhost:5000/cleanup
```

## Development

### Running locally without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run dispatcher
python -m app.api.dispatcher

# Run agent service (in another terminal)
USER_ID=test_user python -m app.api.agent_service
```

### Running tests

```bash
# Use the Jupyter notebook for interactive testing
jupyter notebook test_agent.ipynb
```

## Troubleshooting

### Containers not starting
- Check Docker daemon is running
- Verify Docker socket is accessible: `/var/run/docker.sock`
- Check dispatcher logs: `docker logs agent-dispatcher`

### Redis connection errors
- Ensure Redis container is healthy: `docker ps`
- Check network connectivity: `docker network inspect agent-network`

### LLM errors
- Verify AWS credentials are set correctly
- Check model name is valid for your provider
- Ensure you have access to the specified model

### Memory issues
- Reduce `MAX_CONTAINERS` if running on limited resources
- Decrease `INACTIVE_HOURS` to clean up containers more frequently

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for ReAct agent implementation
- Uses [LangChain](https://github.com/langchain-ai/langchain) for LLM orchestration
- Container orchestration with Docker and Redis