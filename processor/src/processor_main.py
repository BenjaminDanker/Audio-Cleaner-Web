import os
import json
import time
import asyncio
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from video_handler import VideoProcessor
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioCleanerProcessor:
    def __init__(self):
        # Only initialize DefaultAzureCredential if explicitly requested (e.g., in Azure)
        self.credential = None
        if os.getenv('USE_MANAGED_IDENTITY', '').lower() == 'true':
            from azure.identity import DefaultAzureCredential
            self.credential = DefaultAzureCredential()
            logger.info("Using DefaultAzureCredential for Azure authentication.")
        else:
            logger.info("Skipping DefaultAzureCredential; using connection strings for local development.")
        
        # Service Bus
        self.service_bus_connection = os.getenv('AZURE_SERVICE_BUS_CONNECTION_STRING')
        self.queue_name = 'audio-processing-queue'  # Updated queue name
        
        # Storage
        self.storage_connection = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
        self.uploads_container = 'uploads'  # Container for input files
        self.processed_container = 'processed'  # Container for output files
        
        # Cosmos DB
        self.cosmos_connection = os.getenv('COSMOS_CONNECTION_STRING')  # Updated env var name
        self.cosmos_client = CosmosClient.from_connection_string(self.cosmos_connection)
        self.database = self.cosmos_client.get_database_client('audiocleaner')  # Updated database name
        self.jobs_container = self.database.get_container_client('jobs')
        
        logger.info("AudioCleanerProcessor initialized")

    async def process_messages(self):
        """Main message processing loop"""
        async with ServiceBusClient.from_connection_string(
            self.service_bus_connection
        ) as client:
            
            receiver = client.get_queue_receiver(queue_name=self.queue_name)
            
            logger.info(f"Starting to listen for messages on queue: {self.queue_name}")
            
            async with receiver:
                while True:
                    try:
                        # Receive messages
                        received_msgs = await receiver.receive_messages(max_message_count=1, max_wait_time=30)
                        
                        for msg in received_msgs:
                            try:
                                # Extract message body from Service Bus message
                                # The body is a generator, so we need to collect all bytes
                                body_parts = []
                                for part in msg.body:
                                    body_parts.append(part)
                                
                                # Join all parts and decode to string
                                message_body = b''.join(body_parts).decode('utf-8')
                                
                                # Parse the JSON message
                                message_data = json.loads(message_body)
                                
                                # Ensure we have a dictionary
                                if isinstance(message_data, str):
                                    # If it's still a string, try parsing again
                                    message_data = json.loads(message_data)
                                
                                logger.info(f"Processing job: {message_data.get('jobId')}")
                                
                                # Process the video
                                await self.process_video_job(message_data)
                                
                                # Complete the message
                                await receiver.complete_message(msg)
                                logger.info(f"Message completed: {message_data.get('jobId')}")
                                
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                                logger.error(traceback.format_exc())
                                
                                # Update job status to failed
                                if 'message_data' in locals():
                                    await self.update_job_status(
                                        message_data.get('jobId'),
                                        'failed',
                                        error_message=str(e)
                                    )
                                
                                # Dead letter the message
                                await receiver.dead_letter_message(msg, reason="ProcessingError", error_description=str(e))
                    
                    except Exception as e:
                        logger.error(f"Error in message loop: {e}")
                        await asyncio.sleep(5)  # Wait before retrying

    async def process_video_job(self, job_data):
        """Process a single video job"""
        job_id = job_data['jobId']
        user_id = job_data['userId']  # Get user ID from job data
        file_url = job_data['fileUrl']  # Changed from inputBlobUrl
        file_name = job_data['fileName']
        atten_db = job_data.get('attenuation', 30)  # Default attenuation
        
        # Generate output filename with user ID directory structure
        input_path = Path(file_name)
        output_file_name = f"{user_id}/{input_path.stem}_denoised{input_path.suffix}"
        
        # Update job status to processing
        await self.update_job_status(job_id, 'processing', progress=10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download input video from blob storage
                await self.update_job_status(job_id, 'processing', progress=15)
                input_file_path = await self.download_blob_from_url(file_url, temp_dir, job_id)
                await self.update_job_status(job_id, 'processing', progress=25)
                
                # Create a mock uploaded file object for VideoProcessor
                await self.update_job_status(job_id, 'processing', progress=30)
                class MockUploadedFile:
                    def __init__(self, file_path):
                        self.path = file_path
                        self.filename = Path(file_path).name
                    
                    def save(self, target_path):
                        # VideoProcessor expects to save the file to a specific path
                        # If the target path is different from our current path, copy it
                        if os.path.abspath(self.path) != os.path.abspath(target_path):
                            import shutil
                            # Ensure the target directory exists
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            shutil.copy2(self.path, target_path)
                        # If they're the same, do nothing (file is already where it needs to be)
                
                mock_file = MockUploadedFile(input_file_path)
                
                logger.info(f"Starting video processing for job {job_id}")
                
                # Process video using existing VideoProcessor
                await self.update_job_status(job_id, 'processing', progress=35)
                processor = VideoProcessor(mock_file, atten_db)
                
                try:
                    logger.info(f"Calling processor.process() for job {job_id}")
                    await self.update_job_status(job_id, 'processing', progress=40)
                    
                    # Add progress updates during processing
                    import asyncio
                    
                    # Start processing in a separate task and update progress
                    async def process_with_progress():
                        # Simulate progress updates during processing
                        for i in range(45, 75, 5):
                            await asyncio.sleep(2)  # Update every 2 seconds
                            await self.update_job_status(job_id, 'processing', progress=i)
                    
                    # Start progress updates
                    progress_task = asyncio.create_task(process_with_progress())
                    
                    # Run the actual processing
                    loop = asyncio.get_event_loop()
                    output_path = await loop.run_in_executor(None, processor.process)
                    
                    # Cancel progress updates
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    
                    await self.update_job_status(job_id, 'processing', progress=80)
                    logger.info(f"Video processing completed for job {job_id}, output: {output_path}")
                    
                    # Upload processed video to blob storage
                    logger.info(f"Uploading processed video for job {job_id}")
                    await self.update_job_status(job_id, 'processing', progress=85)
                    output_blob_url = await self.upload_blob_with_progress(output_path, output_file_name, job_id)
                    logger.info(f"Upload completed for job {job_id}, URL: {output_blob_url}")
                    
                    await self.update_job_status(job_id, 'processing', progress=95)
                    
                    # Update job status to completed
                    await self.update_job_status(
                        job_id,
                        'completed',
                        progress=100,
                        output_blob_url=output_blob_url,
                        downloadUrl=output_blob_url  # Also set downloadUrl for frontend compatibility
                    )
                    
                    # Delete the input blob now that processing is complete
                    try:
                        await self.delete_input_blob(file_url)
                        logger.info(f"Deleted input blob for job {job_id}")
                    except Exception as delete_error:
                        logger.warning(f"Could not delete input blob for job {job_id}: {delete_error}")
                        # Don't fail the job if we can't delete the input blob
                    
                    logger.info(f"Job {job_id} completed successfully")
                    
                finally:
                    # Cleanup processor temp files
                    processor.immediate_cleanup(logger)
                    
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}")
                logger.error(traceback.format_exc())
                
                # Update job status to failed
                await self.update_job_status(
                    job_id,
                    'failed',
                    error_message=str(e)
                )
                raise

    async def download_blob_from_url(self, blob_url, temp_dir, job_id=None):
        """Download a blob from URL to local file using parallel downloads for large files"""
        try:
            # Parse the blob URL to extract container and blob name
            from urllib.parse import urlparse
            parsed_url = urlparse(blob_url)
            
            # URL format: https://storage.blob.core.windows.net/container/blob/path
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 2:
                raise ValueError(f"Invalid blob URL format: {blob_url}")
                
            container_name = path_parts[0]
            blob_name = '/'.join(path_parts[1:])  # Join remaining parts as blob name can contain /
            
            logger.info(f"Downloading from container: {container_name}, blob: {blob_name}")
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Get blob properties to determine size
            blob_properties = blob_client.get_blob_properties()
            blob_size = blob_properties.size
            
            # Download to temp file
            local_file_path = os.path.join(temp_dir, blob_name)
            
            # Create intermediate directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Use parallel download for files larger than 32MB
            if blob_size > 32 * 1024 * 1024:  # 32MB threshold
                await self._download_large_blob_parallel(blob_client, local_file_path, blob_size, job_id)
            else:
                # For smaller files, use simple download
                with open(local_file_path, 'wb') as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
            
            logger.info(f"Downloaded blob to: {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            raise

    async def _download_large_blob_parallel(self, blob_client, local_file_path, blob_size, job_id=None):
        """Download large blob using parallel range requests with Azure best practices"""
        import concurrent.futures
        import threading
        import time
        import random
        
        # Chunk size (4MB per chunk - optimal for Azure Blob Storage)
        chunk_size = 4 * 1024 * 1024
        chunk_count = (blob_size + chunk_size - 1) // chunk_size
        
        logger.info(f"Downloading large blob ({blob_size} bytes) using {chunk_count} parallel chunks")
        
        # Track download progress
        completed_chunks = 0
        progress_lock = threading.Lock()
        
        # Pre-allocate file with correct size
        with open(local_file_path, 'wb') as f:
            f.truncate(blob_size)
        
        # Store chunks in memory first to avoid file I/O race conditions
        chunks_data = {}
        chunks_lock = threading.Lock()
        
        # Function to download a single chunk with retry logic
        def download_chunk_with_retry(chunk_index, max_retries=3):
            nonlocal completed_chunks
            
            start_offset = chunk_index * chunk_size
            end_offset = min(start_offset + chunk_size - 1, blob_size - 1)
            
            for attempt in range(max_retries + 1):
                try:
                    # Download this range with timeout
                    download_stream = blob_client.download_blob(
                        offset=start_offset, 
                        length=end_offset - start_offset + 1,
                        timeout=60  # 60 second timeout per chunk
                    )
                    chunk_data = download_stream.readall()
                    
                    # Store chunk data thread-safely
                    with chunks_lock:
                        chunks_data[chunk_index] = chunk_data
                    
                    # Update progress thread-safely
                    with progress_lock:
                        completed_chunks += 1
                        progress_percent = (completed_chunks / chunk_count) * 100
                        if completed_chunks % max(1, chunk_count // 10) == 0 or completed_chunks == chunk_count:
                            logger.info(f"Download progress: {completed_chunks}/{chunk_count} chunks ({progress_percent:.1f}%)")
                    
                    return chunk_index
                    
                except Exception as e:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Chunk {chunk_index} download attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Chunk {chunk_index} download failed after {max_retries + 1} attempts: {e}")
                        raise
        
        # Adaptive concurrency based on file size and Azure limits
        max_workers = min(8, max(2, chunk_count // 4))  # 2-8 workers based on chunk count
        
        # Download chunks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk download tasks
            future_to_chunk = {
                executor.submit(download_chunk_with_retry, i): i 
                for i in range(chunk_count)
            }
            
            # Wait for all downloads to complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    future.result()  # Will raise exception if chunk download failed
                except Exception as e:
                    logger.error(f"Fatal error downloading chunk {chunk_index}: {e}")
                    raise
        
        # Write all chunks to file in order (avoiding race conditions)
        logger.info("Writing downloaded chunks to file...")
        with open(local_file_path, 'r+b') as f:
            for chunk_index in range(chunk_count):
                if chunk_index in chunks_data:
                    offset = chunk_index * chunk_size
                    f.seek(offset)
                    f.write(chunks_data[chunk_index])
                else:
                    raise RuntimeError(f"Missing chunk data for chunk {chunk_index}")
        
        logger.info(f"Successfully downloaded {chunk_count} chunks in parallel")

    async def upload_blob(self, local_file_path, blob_name):
        """Upload a local file to blob storage using parallel block uploads"""
        return await self.upload_blob_with_progress(local_file_path, blob_name, None)
    
    async def upload_blob_with_progress(self, local_file_path, blob_name, job_id=None):
        """Upload a local file to blob storage using parallel block uploads with progress tracking"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.processed_container,
                blob=blob_name
            )
            
            # Get file size to determine if we should use block upload
            file_size = os.path.getsize(local_file_path)
            
            # Use parallel block upload for files larger than 64MB, otherwise use simple upload
            if file_size > 64 * 1024 * 1024:  # 64MB threshold
                await self._upload_large_blob_parallel(blob_client, local_file_path, file_size, job_id)
            else:
                # For smaller files, use simple upload
                with open(local_file_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"Uploaded blob: {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading blob: {e}")
            raise

    async def _upload_large_blob_parallel(self, blob_client, local_file_path, file_size, job_id=None):
        """Upload large file using parallel block uploads with Azure best practices"""
        import base64
        import concurrent.futures
        import threading
        import time
        import random
        
        # Block size (4MB per block - optimal for Azure)
        block_size = 4 * 1024 * 1024
        block_count = (file_size + block_size - 1) // block_size
        
        logger.info(f"Uploading large file ({file_size} bytes) using {block_count} blocks in parallel")
        
        # Generate block IDs with proper encoding
        block_ids = []
        for i in range(block_count):
            # Use URL-safe base64 encoding for block IDs
            block_id = base64.urlsafe_b64encode(f"block-{i:06d}".encode()).decode().rstrip('=')
            block_ids.append(block_id)
        
        # Track upload progress with thread safety
        completed_blocks = 0
        progress_lock = threading.Lock()
        uploaded_block_ids = []
        upload_lock = threading.Lock()
        
        # Pre-read file data to avoid repeated I/O
        file_chunks = {}
        logger.info("Pre-reading file chunks...")
        with open(local_file_path, 'rb') as f:
            for i in range(block_count):
                start_offset = i * block_size
                end_offset = min(start_offset + block_size, file_size)
                f.seek(start_offset)
                file_chunks[i] = f.read(end_offset - start_offset)
        
        # Function to upload a single block with retry logic
        def upload_block_with_retry(block_index, max_retries=3):
            nonlocal completed_blocks
            
            block_id = block_ids[block_index]
            block_data = file_chunks[block_index]
            
            for attempt in range(max_retries + 1):
                try:
                    # Upload block with timeout
                    blob_client.stage_block(block_id, block_data, timeout=120)  # 2 minute timeout
                    
                    # Update progress thread-safely
                    with progress_lock:
                        completed_blocks += 1
                        progress_percent = (completed_blocks / block_count) * 100
                        if completed_blocks % max(1, block_count // 10) == 0 or completed_blocks == block_count:
                            logger.info(f"Upload progress: {completed_blocks}/{block_count} blocks ({progress_percent:.1f}%)")
                    
                    # Track uploaded blocks for commit
                    with upload_lock:
                        uploaded_block_ids.append((block_index, block_id))
                    
                    return block_id
                    
                except Exception as e:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Block {block_index} upload attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Block {block_index} upload failed after {max_retries + 1} attempts: {e}")
                        raise
        
        # Adaptive concurrency based on file size
        max_workers = min(8, max(2, block_count // 4))  # 2-8 workers
        
        try:
            # Upload blocks in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all block upload tasks
                future_to_block = {
                    executor.submit(upload_block_with_retry, i): i 
                    for i in range(block_count)
                }
                
                # Wait for all uploads to complete
                for future in concurrent.futures.as_completed(future_to_block):
                    block_index = future_to_block[future]
                    try:
                        future.result()  # Will raise exception if block upload failed
                    except Exception as e:
                        logger.error(f"Fatal error uploading block {block_index}: {e}")
                        # Cancel remaining uploads and cleanup
                        for f in future_to_block:
                            f.cancel()
                        raise
            
            # Sort block IDs by their original order for commit
            uploaded_block_ids.sort(key=lambda x: x[0])
            ordered_block_ids = [block_id for _, block_id in uploaded_block_ids]
            
            # Verify all blocks were uploaded
            if len(ordered_block_ids) != block_count:
                raise RuntimeError(f"Upload incomplete: {len(ordered_block_ids)}/{block_count} blocks uploaded")
            
            # Commit the block list with retry
            logger.info("Committing block list...")
            for attempt in range(3):
                try:
                    blob_client.commit_block_list(ordered_block_ids, timeout=60)
                    logger.info(f"Successfully committed {len(ordered_block_ids)} blocks")
                    break
                except Exception as e:
                    if attempt < 2:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Block list commit attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Block list commit failed after 3 attempts: {e}")
                        raise
                        
        except Exception as e:
            # Cleanup on failure - attempt to delete any uploaded blocks
            logger.error(f"Upload failed, attempting cleanup: {e}")
            try:
                blob_client.delete_blob()
                logger.info("Cleaned up partial upload")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")
            raise

    async def update_job_status(self, job_id, status, **kwargs):
        """Update job status in Cosmos DB"""
        try:
            # Read current job
            job = self.jobs_container.read_item(item=job_id, partition_key=job_id)
            
            # Update status
            job['status'] = status
            job['updatedAt'] = datetime.now().isoformat()
            
            if status == 'processing':
                job['startedAt'] = datetime.now().isoformat()
            elif status in ['completed', 'failed']:
                job['completedAt'] = datetime.now().isoformat()
            
            # Add any additional fields
            for key, value in kwargs.items():
                job[key] = value
            
            # Update in database
            self.jobs_container.upsert_item(job)
            
            logger.info(f"Updated job {job_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            # Don't raise here as it's not critical for processing

    async def delete_input_blob(self, blob_url):
        """Delete the input blob from storage"""
        try:
            # Parse the blob URL to extract container and blob name
            from urllib.parse import urlparse
            parsed_url = urlparse(blob_url)
            
            # URL format: https://storage.blob.core.windows.net/container/blob/path
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 2:
                raise ValueError(f"Invalid blob URL format: {blob_url}")
                
            container_name = path_parts[0]
            blob_name = '/'.join(path_parts[1:])  # Join remaining parts as blob name can contain /
            
            # Get blob client and delete (delete_blob is not async, so no await)
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.delete_blob()
            logger.info(f"Successfully deleted input blob: {blob_name}")
            
        except Exception as e:
            logger.error(f"Error deleting input blob: {e}")
            raise

async def main():
    """Main entry point"""
    processor = AudioCleanerProcessor()
    
    # Start processing messages
    await processor.process_messages()

if __name__ == "__main__":
    asyncio.run(main())
