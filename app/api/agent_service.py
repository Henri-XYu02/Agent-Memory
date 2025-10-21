"""
Flask-based agent service that runs inside Docker containers.
Each container serves a single user's agent requests.
"""

from flask import Flask, request, jsonify
import os
import logging
from typing import Dict, Any
from app.agents.langgraph_react_agent import LangGraphReActAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global agent instance (one per container)
agent = None
user_id = None


def initialize_agent():
    """Initialize the agent instance."""
    global agent, user_id

    user_id = os.getenv("USER_ID", "default")
    provider = os.getenv("AGENT_PROVIDER", "bedrock")
    model_name = os.getenv("AGENT_MODEL", "claude-3-sonnet")

    logger.info(f"Initializing agent for user {user_id} with {provider}/{model_name}")

    agent = LangGraphReActAgent(
        model_name=model_name,
        use_memory=True,
        provider=provider
    )

    logger.info(f"Agent initialized for user {user_id}")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "user_id": user_id,
        "agent_initialized": agent is not None
    }), 200


@app.route("/run", methods=["POST"])
def run_agent():
    """Run the agent with a user message.

    Expected JSON payload:
    {
        "message": "user message",
        "session_id": "optional session id",
        "system_prompt": "optional system prompt"
    }
    """
    if agent is None:
        return jsonify({"error": "Agent not initialized"}), 500

    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        message = data["message"]
        session_id = data.get("session_id", "default")
        system_prompt = data.get("system_prompt")

        logger.info(f"Running agent for user {user_id}, session {session_id}")

        # Create thread_id combining user_id and session_id
        thread_id = f"{user_id}_{session_id}"

        # Run the agent
        response = agent.run(
            message=message,
            thread_id=thread_id,
            system_prompt=system_prompt
        )

        # Extract the final message from the response
        messages = response.get("messages", [])
        if messages:
            final_message = messages[-1]
            result = {
                "user_id": user_id,
                "session_id": session_id,
                "response": final_message.content if hasattr(final_message, 'content') else str(final_message),
                "message_count": len(messages)
            }
        else:
            result = {
                "user_id": user_id,
                "session_id": session_id,
                "response": "No response generated",
                "message_count": 0
            }

        logger.info(f"Agent completed for user {user_id}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """Get the status of this agent container."""
    return jsonify({
        "user_id": user_id,
        "agent_initialized": agent is not None,
        "provider": os.getenv("AGENT_PROVIDER", "bedrock"),
        "model": os.getenv("AGENT_MODEL", "claude-3-sonnet")
    }), 200


def create_app():
    """Create and configure the Flask app."""
    initialize_agent()
    return app


if __name__ == "__main__":
    initialize_agent()
    port = int(os.getenv("FLASK_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
