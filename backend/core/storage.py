import boto3
import logging
from pathlib import Path
from typing import Optional
from datetime import timedelta

from .config import settings

logger = logging.getLogger(__name__)


class S3Storage:
    """S3/MinIO storage client wrapper."""

    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )
        self.bucket = settings.S3_BUCKET

    async def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload file from local path to S3, return S3 URL."""
        try:
            self.client.upload_file(local_path, self.bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
            return self._get_s3_url(s3_key)
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    async def download_file(self, s3_key: str, local_path: str) -> str:
        """Download file from S3 to local path, return local path."""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket}/{s3_key} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def generate_presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for file access."""
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            raise

    def _get_s3_url(self, s3_key: str) -> str:
        """Get HTTP URL for S3 object."""
        if settings.S3_USE_SSL:
            return f"https://{self.bucket}.s3.{settings.S3_REGION}.amazonaws.com/{s3_key}"
        else:
            # For MinIO local development
            return f"{settings.S3_ENDPOINT}/{self.bucket}/{s3_key}"

    async def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            raise


# Singleton instance
storage = S3Storage()
