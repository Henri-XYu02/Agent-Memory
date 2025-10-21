"""
Central dispatcher service that manages user containers and routes requests.
Includes heartbeat monitoring and LRU eviction.
"""

from flask import Flask, request, jsonify
import requests
import logging
import threading
import time
from typing import Dict, Any, Optional
from app.container.redis_manager import RedisContainerManager
from app.container.docker_manager import DockerContainerManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global managers
redis_manager = None
docker_manager = None
heartbeat_thread = None
cleanup_thread = None


def initialize_managers():
    """Initialize Redis and Docker managers."""
    global redis_manager, docker_manager

    # Get configuration from environment
    import os
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    max_containers = int(os.getenv("MAX_CONTAINERS", 5))
    agent_image = os.getenv("AGENT_IMAGE", "agent-memory:latest")

    logger.info(f"Initializing managers with Redis at {redis_host}:{redis_port}")

    redis_manager = RedisContainerManager(
        redis_host=redis_host,
        redis_port=redis_port
    )

    docker_manager = DockerContainerManager(
        redis_manager=redis_manager,
        max_containers=max_containers,
        agent_image=agent_image
    )

    logger.info("Managers initialized successfully")


def heartbeat_monitor():
    """Background thread to perform heartbeat checks on containers."""
    logger.info("Heartbeat monitor started")

    while True:
        try:
            time.sleep(60)  # Check every minute

            containers = redis_manager.get_all_containers()
            logger.info(f"Heartbeat check: {len(containers)} containers active")

            for container_info in containers:
                container_ip = container_info["ip"]
                container_port = container_info["port"]
                container_id = container_info["container_id"]

                try:
                    # Send health check request
                    response = requests.get(
                        f"http://{container_ip}:{container_port}/health",
                        timeout=5
                    )

                    if response.status_code == 200:
                        logger.debug(f"Container {container_id[:12]} is healthy")
                    else:
                        logger.warning(f"Container {container_id[:12]} returned status {response.status_code}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"Heartbeat failed for container {container_id[:12]}: {e}")
                    # Container might be dead - it will be cleaned up by cleanup_inactive_containers

        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}", exc_info=True)


def cleanup_inactive_containers():
    """Background thread to clean up inactive containers."""
    logger.info("Cleanup monitor started")

    while True:
        try:
            # Check every 30 minutes
            time.sleep(1800)

            inactive_hours = int(os.getenv("INACTIVE_HOURS", 6))
            logger.info(f"Running cleanup for containers inactive for {inactive_hours} hours")

            count = docker_manager.cleanup_inactive_containers(inactive_hours)
            logger.info(f"Cleaned up {count} inactive containers")

        except Exception as e:
            logger.error(f"Error in cleanup monitor: {e}", exc_info=True)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "dispatcher",
        "active_containers": redis_manager.get_active_container_count() if redis_manager else 0
    }), 200


@app.route("/agent/run", methods=["POST"])
def run_agent():
    """Main endpoint to run agent for a user.

    Expected JSON payload:
    {
        "user_id": "user123",
        "message": "user message",
        "session_id": "optional session id",
        "system_prompt": "optional system prompt"
    }
    """
    try:
        data = request.get_json()

        if not data or "user_id" not in data or "message" not in data:
            return jsonify({"error": "Missing required fields: user_id, message"}), 400

        user_id = data["user_id"]
        message = data["message"]
        session_id = data.get("session_id", "default")
        system_prompt = data.get("system_prompt")

        logger.info(f"Request received for user {user_id}")

        # Get or create container for user
        container_ip, container_port = docker_manager.get_or_create_container(user_id)

        logger.info(f"Routing request to container at {container_ip}:{container_port}")

        # Forward request to container
        response = requests.post(
            f"http://{container_ip}:{container_port}/run",
            json={
                "message": message,
                "session_id": session_id,
                "system_prompt": system_prompt
            },
            timeout=300  # 5 minute timeout for agent execution
        )

        # Update last active timestamp
        redis_manager.update_last_active(user_id)

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": "Agent execution failed",
                "details": response.text
            }), response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with container: {e}")
        return jsonify({"error": f"Container communication failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error in run_agent: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/containers", methods=["GET"])
def list_containers():
    """List all active containers."""
    try:
        containers = redis_manager.get_all_containers()
        return jsonify({
            "count": len(containers),
            "containers": containers
        }), 200
    except Exception as e:
        logger.error(f"Error listing containers: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/containers/<user_id>", methods=["GET"])
def get_container_status(user_id: str):
    """Get status of a specific user's container."""
    try:
        status = docker_manager.get_container_status(user_id)
        if status:
            return jsonify(status), 200
        else:
            return jsonify({"error": "No container found for user"}), 404
    except Exception as e:
        logger.error(f"Error getting container status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/containers/<user_id>", methods=["DELETE"])
def stop_user_container(user_id: str):
    """Stop a specific user's container."""
    try:
        container_info = redis_manager.get_container_for_user(user_id)
        if not container_info:
            return jsonify({"error": "No container found for user"}), 404

        docker_manager.stop_container(container_info["container_id"])
        return jsonify({"message": f"Container for user {user_id} stopped"}), 200
    except Exception as e:
        logger.error(f"Error stopping container: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/cleanup", methods=["POST"])
def trigger_cleanup():
    """Manually trigger cleanup of inactive containers."""
    try:
        data = request.get_json() or {}
        inactive_hours = data.get("inactive_hours", 6)

        count = docker_manager.cleanup_inactive_containers(inactive_hours)
        return jsonify({
            "message": f"Cleaned up {count} inactive containers",
            "count": count
        }), 200
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
        return jsonify({"error": str(e)}), 500


def start_background_tasks():
    """Start background monitoring threads."""
    global heartbeat_thread, cleanup_thread

    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()

    cleanup_thread = threading.Thread(target=cleanup_inactive_containers, daemon=True)
    cleanup_thread.start()

    logger.info("Background tasks started")


def create_app():
    """Create and configure the Flask app."""
    import os

    initialize_managers()
    start_background_tasks()
    return app


if __name__ == "__main__":
    import os

    initialize_managers()
    start_background_tasks()

    port = int(os.getenv("DISPATCHER_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
