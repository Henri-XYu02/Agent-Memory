import redis
import json
import time
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class RedisContainerManager:
    """Manages container metadata and state using Redis."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        """Initialize Redis connection.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.container_prefix = "container:"
        self.user_container_key = "user_containers"  # Hash: user_id -> container_id
        self.container_info_prefix = "container_info:"  # Hash: container metadata
        self.lru_key = "container_lru"  # Sorted set for LRU tracking

    def register_container(self, user_id: str, container_id: str, container_ip: str, container_port: int) -> None:
        """Register a new container for a user.

        Args:
            user_id: User identifier
            container_id: Docker container ID
            container_ip: Container IP address
            container_port: Container port
        """
        # Store user -> container mapping
        self.redis_client.hset(self.user_container_key, user_id, container_id)

        # Store container info
        container_info = {
            "container_id": container_id,
            "user_id": user_id,
            "ip": container_ip,
            "port": container_port,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "status": "active"
        }
        self.redis_client.hset(
            f"{self.container_info_prefix}{container_id}",
            mapping=container_info
        )

        # Add to LRU tracking (sorted set with timestamp as score)
        self.redis_client.zadd(self.lru_key, {container_id: time.time()})

    def get_container_for_user(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get container information for a user.

        Args:
            user_id: User identifier

        Returns:
            Container info dict or None if no container exists
        """
        container_id = self.redis_client.hget(self.user_container_key, user_id)
        if not container_id:
            return None

        container_info = self.redis_client.hgetall(f"{self.container_info_prefix}{container_id}")
        if not container_info:
            # Clean up orphaned reference
            self.redis_client.hdel(self.user_container_key, user_id)
            return None

        return container_info

    def update_last_active(self, user_id: str) -> None:
        """Update the last active timestamp for a user's container.

        Args:
            user_id: User identifier
        """
        container_id = self.redis_client.hget(self.user_container_key, user_id)
        if container_id:
            now = datetime.now().isoformat()
            self.redis_client.hset(
                f"{self.container_info_prefix}{container_id}",
                "last_active",
                now
            )
            # Update LRU score
            self.redis_client.zadd(self.lru_key, {container_id: time.time()})

    def get_active_container_count(self) -> int:
        """Get the number of active containers.

        Returns:
            Number of active containers
        """
        return self.redis_client.zcard(self.lru_key)

    def get_least_recently_used_container(self) -> Optional[Dict[str, str]]:
        """Get the least recently used container.

        Returns:
            Container info dict or None if no containers exist
        """
        # Get container with lowest score (oldest timestamp)
        result = self.redis_client.zrange(self.lru_key, 0, 0, withscores=True)
        if not result:
            return None

        container_id, _ = result[0]
        container_info = self.redis_client.hgetall(f"{self.container_info_prefix}{container_id}")
        return container_info if container_info else None

    def get_inactive_containers(self, inactive_hours: int = 6) -> List[Dict[str, str]]:
        """Get containers that have been inactive for specified hours.

        Args:
            inactive_hours: Number of hours of inactivity threshold

        Returns:
            List of container info dicts
        """
        inactive_containers = []
        threshold = datetime.now() - timedelta(hours=inactive_hours)

        # Get all container IDs
        all_container_ids = self.redis_client.zrange(self.lru_key, 0, -1)

        for container_id in all_container_ids:
            container_info = self.redis_client.hgetall(f"{self.container_info_prefix}{container_id}")
            if not container_info:
                continue

            last_active_str = container_info.get("last_active")
            if last_active_str:
                last_active = datetime.fromisoformat(last_active_str)
                if last_active < threshold:
                    inactive_containers.append(container_info)

        return inactive_containers

    def remove_container(self, container_id: str) -> None:
        """Remove a container from tracking.

        Args:
            container_id: Docker container ID
        """
        # Get container info to find user_id
        container_info = self.redis_client.hgetall(f"{self.container_info_prefix}{container_id}")
        if container_info:
            user_id = container_info.get("user_id")
            if user_id:
                self.redis_client.hdel(self.user_container_key, user_id)

        # Remove container info
        self.redis_client.delete(f"{self.container_info_prefix}{container_id}")

        # Remove from LRU tracking
        self.redis_client.zrem(self.lru_key, container_id)

    def get_all_containers(self) -> List[Dict[str, str]]:
        """Get all active containers.

        Returns:
            List of container info dicts
        """
        all_container_ids = self.redis_client.zrange(self.lru_key, 0, -1)
        containers = []

        for container_id in all_container_ids:
            container_info = self.redis_client.hgetall(f"{self.container_info_prefix}{container_id}")
            if container_info:
                containers.append(container_info)

        return containers

    def clear_all(self) -> None:
        """Clear all container data (for testing purposes)."""
        # Get all container IDs
        all_container_ids = self.redis_client.zrange(self.lru_key, 0, -1)

        # Delete all container info
        for container_id in all_container_ids:
            self.redis_client.delete(f"{self.container_info_prefix}{container_id}")

        # Delete tracking structures
        self.redis_client.delete(self.user_container_key)
        self.redis_client.delete(self.lru_key)
