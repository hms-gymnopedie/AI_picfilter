import logging
import tempfile
from pathlib import Path
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


def create_temp_directory(prefix: str = "picfilter") -> Path:
    """Create a temporary directory for ML task operations."""
    temp_dir = Path(tempfile.gettempdir()) / prefix / str(datetime.utcnow().timestamp())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_directory(temp_dir: Path):
    """Clean up temporary directory and its contents."""
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


def update_job_progress_redis(job_id: UUID, progress: float, redis_client):
    """Update job progress in Redis for real-time tracking."""
    try:
        key = f"job_progress:{job_id}"
        redis_client.setex(key, 3600, str(progress))  # 1 hour TTL
        logger.debug(f"Updated job progress: {job_id} = {progress}")
    except Exception as e:
        logger.error(f"Failed to update job progress in Redis: {e}")


def get_job_progress_redis(job_id: UUID, redis_client) -> float:
    """Get job progress from Redis."""
    try:
        key = f"job_progress:{job_id}"
        progress = redis_client.get(key)
        return float(progress) if progress else 0.0
    except Exception as e:
        logger.error(f"Failed to get job progress from Redis: {e}")
        return 0.0


def progress_callback(job_id: UUID, current: float, total: float, redis_client):
    """Callback function for ML tasks to report progress."""
    progress = (current / total) if total > 0 else 0.0
    update_job_progress_redis(job_id, min(progress, 1.0), redis_client)
