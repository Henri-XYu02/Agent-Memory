import docker
import time
import logging
from typing import Optional, Dict, Tuple
from app.container.redis_manager import RedisContainerManager

logger = logging.getLogger(__name__)


class DockerContainerManager:
    """Manages Docker containers for user isolation with LRU eviction."""

    def __init__(
        self,
        redis_manager: RedisContainerManager,
        max_containers: int = 5,
        agent_image: str = "agent-memory:latest",
        network_name: str = "agent-network"
    ):
        """Initialize Docker container manager.

        Args:
            redis_manager: RedisContainerManager instance
            max_containers: Maximum number of containers allowed
            agent_image: Docker image name for agent containers
            network_name: Docker network name
        """
        self.docker_client = docker.from_env()
        self.redis_manager = redis_manager
        self.max_containers = max_containers
        self.agent_image = agent_image
        self.network_name = network_name

        # Ensure network exists
        self._ensure_network()

    def _ensure_network(self) -> None:
        """Ensure the Docker network exists."""
        try:
            self.docker_client.networks.get(self.network_name)
            logger.info(f"Network {self.network_name} already exists")
        except docker.errors.NotFound:
            self.docker_client.networks.create(self.network_name, driver="bridge")
            logger.info(f"Created network {self.network_name}")

    def _evict_lru_container(self) -> None:
        """Evict the least recently used container."""
        lru_container = self.redis_manager.get_least_recently_used_container()
        if not lru_container:
            logger.warning("No LRU container found to evict")
            return

        container_id = lru_container["container_id"]
        logger.info(f"Evicting LRU container {container_id} for user {lru_container['user_id']}")
        self.stop_container(container_id)

    def get_or_create_container(self, user_id: str) -> Tuple[str, int]:
        """Get existing container or create a new one for a user.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (container_ip, container_port)
        """
        # Check if container already exists
        container_info = self.redis_manager.get_container_for_user(user_id)
        if container_info:
            # Update last active time
            self.redis_manager.update_last_active(user_id)
            return container_info["ip"], int(container_info["port"])

        # Check if we need to evict a container
        active_count = self.redis_manager.get_active_container_count()
        if active_count >= self.max_containers:
            logger.info(f"Max containers ({self.max_containers}) reached, evicting LRU")
            self._evict_lru_container()

        # Create new container
        return self._create_container(user_id)

    def _create_container(self, user_id: str) -> Tuple[str, int]:
        """Create a new container for a user.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (container_ip, container_port)
        """
        container_name = f"agent-{user_id}"
        agent_port = 5001  # Internal port the Flask app runs on

        logger.info(f"Creating container for user {user_id}")

        try:
            # Start container
            container = self.docker_client.containers.run(
                self.agent_image,
                name=container_name,
                detach=True,
                network=self.network_name,
                environment={
                    "USER_ID": user_id,
                    "FLASK_PORT": str(agent_port)
                },
                labels={
                    "user_id": user_id,
                    "managed_by": "agent-dispatcher"
                },
                # Auto-remove container when stopped
                auto_remove=False
            )

            # Wait a bit for container to start
            time.sleep(2)

            # Get container IP
            container.reload()
            container_ip = container.attrs['NetworkSettings']['Networks'][self.network_name]['IPAddress']

            # Register container in Redis
            self.redis_manager.register_container(
                user_id=user_id,
                container_id=container.id,
                container_ip=container_ip,
                container_port=agent_port
            )

            logger.info(f"Container {container.id[:12]} created for user {user_id} at {container_ip}:{agent_port}")
            return container_ip, agent_port

        except docker.errors.ImageNotFound:
            logger.error(f"Docker image {self.agent_image} not found")
            raise
        except Exception as e:
            logger.error(f"Error creating container for user {user_id}: {e}")
            raise

    def stop_container(self, container_id: str) -> None:
        """Stop and remove a container.

        Args:
            container_id: Docker container ID
        """
        try:
            container = self.docker_client.containers.get(container_id)
            logger.info(f"Stopping container {container_id[:12]}")
            container.stop(timeout=10)
            container.remove()
            logger.info(f"Container {container_id[:12]} stopped and removed")
        except docker.errors.NotFound:
            logger.warning(f"Container {container_id} not found")
        except Exception as e:
            logger.error(f"Error stopping container {container_id}: {e}")
        finally:
            # Remove from Redis tracking
            self.redis_manager.remove_container(container_id)

    def cleanup_inactive_containers(self, inactive_hours: int = 6) -> int:
        """Clean up containers that have been inactive for specified hours.

        Args:
            inactive_hours: Number of hours of inactivity threshold

        Returns:
            Number of containers cleaned up
        """
        inactive_containers = self.redis_manager.get_inactive_containers(inactive_hours)
        count = 0

        for container_info in inactive_containers:
            container_id = container_info["container_id"]
            logger.info(f"Cleaning up inactive container {container_id[:12]}")
            self.stop_container(container_id)
            count += 1

        return count

    def get_container_status(self, user_id: str) -> Optional[Dict]:
        """Get the status of a user's container.

        Args:
            user_id: User identifier

        Returns:
            Container status dict or None
        """
        container_info = self.redis_manager.get_container_for_user(user_id)
        if not container_info:
            return None

        try:
            container = self.docker_client.containers.get(container_info["container_id"])
            return {
                "user_id": user_id,
                "container_id": container.id[:12],
                "status": container.status,
                "created_at": container_info["created_at"],
                "last_active": container_info["last_active"]
            }
        except docker.errors.NotFound:
            # Container doesn't exist but is in Redis - clean up
            self.redis_manager.remove_container(container_info["container_id"])
            return None

    def cleanup_all_containers(self) -> None:
        """Stop and remove all managed containers (for testing/shutdown)."""
        all_containers = self.redis_manager.get_all_containers()

        for container_info in all_containers:
            self.stop_container(container_info["container_id"])

        logger.info(f"Cleaned up {len(all_containers)} containers")
