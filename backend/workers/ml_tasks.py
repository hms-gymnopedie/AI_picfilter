import sys
import logging
import json
from datetime import datetime
from uuid import UUID
from pathlib import Path
import redis

sys.path.insert(0, "/app/src")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import settings
from backend.core.storage import storage
from backend.models import Job, Style
from .utils import create_temp_directory, cleanup_temp_directory, update_job_progress_redis
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Initialize Redis for progress tracking
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Initialize async database
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session():
    async with async_session_maker() as session:
        return session


def progress_callback(job_id: str, current: float, total: float):
    """Callback for ML tasks to report progress."""
    progress = (current / total) if total > 0 else 0.0
    update_job_progress_redis(UUID(job_id), min(progress, 1.0), redis_client)


async def update_job_status(job_id: str, status: str, error_message: str = None, result_key: str = None, model_key: str = None):
    """Update job status in database."""
    session = await get_async_session()
    try:
        result = await session.execute(select(Job).where(Job.id == UUID(job_id)))
        job = result.scalars().first()
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            if result_key:
                job.result_key = result_key
            if status == "processing":
                job.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                job.completed_at = datetime.utcnow()

            # Update style model_key if this is a learn job
            if status == "completed" and model_key:
                result = await session.execute(select(Style).where(Style.id == job.style_id))
                style = result.scalars().first()
                if style:
                    style.model_key = model_key

            await session.commit()
            logger.info(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")
    finally:
        await session.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="ml_tasks.learn_style")
def learn_style(self, job_id: str, style_id: str, reference_image_s3_keys: list, config: dict):
    """
    Learn style from reference images using ML model.

    Args:
        job_id: Job UUID
        style_id: Style UUID
        reference_image_s3_keys: List of S3 keys for reference images
        config: Configuration dict with model_type, strength, etc.
    """
    import asyncio

    temp_dir = None
    try:
        logger.info(f"Starting style learning: job_id={job_id}, style_id={style_id}")

        # Update job status to processing
        asyncio.run(update_job_status(job_id, "processing"))

        # Create temporary directory
        temp_dir = create_temp_directory("style_learning")

        # Download reference images
        reference_image_paths = []
        for i, s3_key in enumerate(reference_image_s3_keys):
            local_path = str(temp_dir / f"ref_{i}.jpg")
            asyncio.run(storage.download_file(s3_key, local_path))
            reference_image_paths.append(local_path)
            # Report progress
            update_job_progress_redis(UUID(job_id), i / len(reference_image_s3_keys) * 0.2, redis_client)

        logger.info(f"Downloaded {len(reference_image_paths)} reference images")

        # Import and run ML inference based on model type
        model_type = config.get("model_type", "nilut")
        update_job_progress_redis(UUID(job_id), 0.3, redis_client)

        try:
            from api.inference_api import run_style_learning

            model_output_path = str(temp_dir / "model.pt")

            # Call ML API with model type
            if model_type == "nilut":
                logger.info("Using NILUT model for style learning")
            elif model_type == "lut3d":
                logger.info("Using Image-Adaptive 3D LUT model for style learning")
            else:
                logger.warning(f"Unknown model type: {model_type}, using default")

            result = run_style_learning(
                reference_images=reference_image_paths,
                output_path=model_output_path,
                model_type=model_type,
                config=config,
                progress_callback=lambda curr, total: progress_callback(job_id, curr, total),
            )

            logger.info(f"Style learning completed: {result}")
        except ImportError:
            logger.warning("inference_api not available, using mock result")
            # Mock result for testing
            model_output_path = str(temp_dir / "model.pt")
            Path(model_output_path).write_text(f"mock_{model_type}_model_data")

        # Upload model to S3
        s3_model_key = f"styles/{style_id}/model.pt"
        asyncio.run(storage.upload_file(model_output_path, s3_model_key))
        logger.info(f"Uploaded model to S3: {s3_model_key}")

        # Update job status to completed
        asyncio.run(update_job_status(job_id, "completed", model_key=s3_model_key, result_key=s3_model_key))
        update_job_progress_redis(UUID(job_id), 1.0, redis_client)

        return {
            "status": "completed",
            "job_id": job_id,
            "style_id": style_id,
            "model_key": s3_model_key,
        }

    except Exception as exc:
        logger.error(f"Style learning failed: {exc}", exc_info=True)
        import asyncio
        asyncio.run(update_job_status(job_id, "failed", error_message=str(exc)))

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        else:
            raise

    finally:
        if temp_dir:
            cleanup_temp_directory(temp_dir)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="ml_tasks.apply_filter")
def apply_filter(self, job_id: str, input_image_s3_key: str, style_id: str, intensity: float):
    """
    Apply learned style filter to image.

    Args:
        job_id: Job UUID
        input_image_s3_key: S3 key of input image
        style_id: Style UUID
        intensity: Filter intensity (0.0 to 1.0)
    """
    import asyncio

    temp_dir = None
    try:
        logger.info(f"Starting filter application: job_id={job_id}, style_id={style_id}")

        # Update job status to processing
        asyncio.run(update_job_status(job_id, "processing"))

        # Create temporary directory
        temp_dir = create_temp_directory("apply_filter")

        # Download input image and style model
        input_image_path = str(temp_dir / "input.jpg")
        asyncio.run(storage.download_file(input_image_s3_key, input_image_path))
        update_job_progress_redis(UUID(job_id), 0.3, redis_client)

        model_s3_key = f"styles/{style_id}/model.pt"
        model_path = str(temp_dir / "model.pt")
        asyncio.run(storage.download_file(model_s3_key, model_path))
        logger.info(f"Downloaded input image and model")
        update_job_progress_redis(UUID(job_id), 0.5, redis_client)

        # Get style model_type from database for logging
        session = await get_async_session()
        result = await session.execute(select(Style).where(Style.id == UUID(style_id)))
        style = result.scalars().first()
        model_type = style.model_type if style else "unknown"
        await session.close()
        logger.info(f"Style model type: {model_type}")

        # Import and run ML inference
        try:
            from api.inference_api import run_apply_filter

            output_image_path = str(temp_dir / "output.jpg")
            result = run_apply_filter(
                input_image=input_image_path,
                model_path=model_path,
                output_path=output_image_path,
                intensity=intensity,
                progress_callback=lambda curr, total: progress_callback(job_id, curr, total),
            )

            logger.info(f"Filter application completed: {result}")
        except ImportError:
            logger.warning("inference_api not available, using mock result")
            # Mock result for testing
            import shutil
            output_image_path = str(temp_dir / "output.jpg")
            shutil.copy(input_image_path, output_image_path)

        # Upload result to S3
        s3_result_key = f"results/{job_id}/output.jpg"
        asyncio.run(storage.upload_file(output_image_path, s3_result_key))
        logger.info(f"Uploaded result to S3: {s3_result_key}")

        # Update job status to completed
        asyncio.run(update_job_status(job_id, "completed", result_key=s3_result_key))
        update_job_progress_redis(UUID(job_id), 1.0, redis_client)

        return {
            "status": "completed",
            "job_id": job_id,
            "style_id": style_id,
            "result_key": s3_result_key,
        }

    except Exception as exc:
        logger.error(f"Filter application failed: {exc}", exc_info=True)
        import asyncio
        asyncio.run(update_job_status(job_id, "failed", error_message=str(exc)))

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        else:
            raise

    finally:
        if temp_dir:
            cleanup_temp_directory(temp_dir)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="ml_tasks.export_cube")
def export_cube(self, job_id: str, style_id: str, lut_size: int):
    """
    Export learned style as .cube LUT file.

    Args:
        job_id: Job UUID
        style_id: Style UUID
        lut_size: LUT grid size (17, 33, or 65)
    """
    import asyncio

    temp_dir = None
    try:
        logger.info(f"Starting .cube export: job_id={job_id}, style_id={style_id}")

        # Update job status to processing
        asyncio.run(update_job_status(job_id, "processing"))

        # Create temporary directory
        temp_dir = create_temp_directory("export_cube")

        # Download style model
        model_s3_key = f"styles/{style_id}/model.pt"
        model_path = str(temp_dir / "model.pt")
        asyncio.run(storage.download_file(model_s3_key, model_path))
        logger.info(f"Downloaded model for .cube export")
        update_job_progress_redis(UUID(job_id), 0.3, redis_client)

        # Get style model_type from database for logging
        session = await get_async_session()
        result = await session.execute(select(Style).where(Style.id == UUID(style_id)))
        style = result.scalars().first()
        model_type = style.model_type if style else "unknown"
        await session.close()
        logger.info(f"Style model type for .cube export: {model_type}")

        # Import and run ML inference
        try:
            from api.inference_api import run_export_cube

            cube_output_path = str(temp_dir / f"style_{lut_size}.cube")
            result = run_export_cube(
                model_path=model_path,
                output_path=cube_output_path,
                lut_size=lut_size,
                progress_callback=lambda curr, total: progress_callback(job_id, curr, total),
            )

            logger.info(f".cube export completed: {result}")
        except ImportError:
            logger.warning("inference_api not available, using mock result")
            # Mock result for testing
            cube_output_path = str(temp_dir / f"style_{lut_size}.cube")
            Path(cube_output_path).write_text(f"TITLE \"Style {style_id}\"\n")

        # Upload .cube file to S3
        s3_cube_key = f"cubes/{style_id}/export_{job_id}.cube"
        asyncio.run(storage.upload_file(cube_output_path, s3_cube_key))
        logger.info(f"Uploaded .cube file to S3: {s3_cube_key}")

        # Update job status to completed
        asyncio.run(update_job_status(job_id, "completed", result_key=s3_cube_key))
        update_job_progress_redis(UUID(job_id), 1.0, redis_client)

        return {
            "status": "completed",
            "job_id": job_id,
            "style_id": style_id,
            "result_key": s3_cube_key,
            "lut_size": lut_size,
        }

    except Exception as exc:
        logger.error(f".cube export failed: {exc}", exc_info=True)
        import asyncio
        asyncio.run(update_job_status(job_id, "failed", error_message=str(exc)))

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)
        else:
            raise

    finally:
        if temp_dir:
            cleanup_temp_directory(temp_dir)
