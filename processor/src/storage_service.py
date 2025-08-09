import os
import logging
from pathlib import Path
from urllib.parse import urlparse
from azure.storage.blob import StandardBlobTier

logger = logging.getLogger(__name__)

def _parse_blob_url(blob_url: str):
    """Return (container, blob_path) for a well-formed https://<acct>.blob.core.windows.net/<container>/<blob> URL.

    Raises ValueError if parsing fails.
    """
    parsed = urlparse(blob_url)
    parts = parsed.path.strip('/').split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid blob URL (expected at least container/blob): {blob_url}")
    return parts[0], '/'.join(parts[1:])

class BlobStorageService:
    def __init__(self, blob_service_client, processed_container: str):
        self._client = blob_service_client
        self._processed = processed_container

    async def download(self, blob_url: str, dest_dir: str, expected_user: str, job_id: str):
        container, blob_name = _parse_blob_url(blob_url)
        # Enforce first path segment EXACT match to expected_user to mitigate prefix bypass (e.g., userX vs userX_evil)
        first_segment = blob_name.split('/', 1)[0]
        if first_segment != expected_user:
            raise ValueError("Security violation: blob does not belong to user")
        local_path = os.path.join(dest_dir, Path(blob_name).name)
        blob_client = self._client.get_blob_client(container=container, blob=blob_name)
        stream = await blob_client.download_blob(max_concurrency=4)
        with open(local_path, 'wb') as f:
            async for chunk in stream.chunks():
                f.write(chunk)
        logger.info(f"Downloaded blob to {local_path}")
        return local_path

    async def upload_processed(self, local_file: str, blob_name: str) -> str:
        blob_client = self._client.get_blob_client(container=self._processed, blob=blob_name)
        size = os.path.getsize(local_file)
        max_conc = 4 if size > 8 * 1024 * 1024 else 2
        # Determine desired tier. Prefer Cold (lower cost) if supported by the installed SDK, else fall back to Cool.
        desired_tier_name = "Cold"
        if not hasattr(StandardBlobTier, desired_tier_name):  # Older SDKs (<12.19.0) won't have Cold
            desired_tier_name = "Cool"
        tier_enum = getattr(StandardBlobTier, desired_tier_name)
        logger.info(f"selected tier {tier_enum}, desired_tier_name {desired_tier_name}")

        with open(local_file, 'rb') as data:
            await blob_client.upload_blob(
                data,
                overwrite=True,
                standard_blob_tier=tier_enum,  # Must be Enum, not raw string, else azure SDK will try .value and fail
                max_concurrency=max_conc
            )
        logger.info(f"Uploaded processed blob {blob_client.url}")
        return blob_client.url

    async def delete(self, blob_url: str):
        container, blob_name = _parse_blob_url(blob_url)
        blob_client = self._client.get_blob_client(container=container, blob=blob_name)
        if not await blob_client.exists():
            logger.warning(f"Blob not found for deletion: {blob_name}")
            return
        await blob_client.delete_blob()
        logger.info(f"Deleted blob {blob_name}")
