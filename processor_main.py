import os
import json
import time
import asyncio
import tempfile
import logging
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
        self.container_name = 'audio-files'  # Updated container name
        
        # Cosmos DB
        self.cosmos_connection = os.getenv('AZURE_COSMOS_CONNECTION_STRING')  # Updated env var name
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
                                # Parse message
                                message_data = json.loads(str(msg))
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
        file_url = job_data['fileUrl']  # Changed from inputBlobUrl
        file_name = job_data['fileName']
        atten_db = job_data.get('attenuation', 30)  # Default attenuation
        
        # Generate output filename
        input_path = Path(file_name)
        output_file_name = f"{input_path.stem}_denoised{input_path.suffix}"
        
        # Update job status to processing
        await self.update_job_status(job_id, 'processing')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download input video from blob storage
                input_file_path = await self.download_blob_from_url(file_url, temp_dir)
                
                # Create a mock uploaded file object for VideoProcessor
                class MockUploadedFile:
                    def __init__(self, file_path):
                        self.path = file_path
                        self.filename = Path(file_path).name
                    
                    def save(self, path):
                        # VideoProcessor expects to save the file, but we already have it
                        import shutil
                        shutil.copy2(self.path, path)
                
                mock_file = MockUploadedFile(input_file_path)
                
                # Process video using existing VideoProcessor
                processor = VideoProcessor(mock_file, atten_db)
                
                try:
                    output_path = processor.process()
                    
                    # Upload processed video to blob storage
                    output_blob_url = await self.upload_blob(output_path, output_file_name)
                    
                    # Update job status to completed
                    await self.update_job_status(
                        job_id,
                        'completed',
                        output_blob_url=output_blob_url
                    )
                    
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

    async def download_blob_from_url(self, blob_url, temp_dir):
        """Download a blob from URL to local file"""
        try:
            # Extract container and blob name from URL
            url_parts = blob_url.split('/')
            container_name = url_parts[-2]
            blob_name = url_parts[-1]
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Download to temp file
            local_file_path = os.path.join(temp_dir, blob_name)
            
            with open(local_file_path, 'wb') as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            logger.info(f"Downloaded blob to: {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            raise

    async def upload_blob(self, local_file_path, blob_name):
        """Upload a local file to blob storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            with open(local_file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            blob_url = blob_client.url
            logger.info(f"Uploaded blob: {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading blob: {e}")
            raise

    async def update_job_status(self, job_id, status, **kwargs):
        """Update job status in Cosmos DB"""
        try:
            # Read current job
            job = self.jobs_container.read_item(item=job_id, partition_key=job_id)
            
            # Update status
            job['status'] = status
            job['updatedAt'] = time.time()
            
            if status == 'processing':
                job['startedAt'] = time.time()
            elif status in ['completed', 'failed']:
                job['completedAt'] = time.time()
            
            # Add any additional fields
            for key, value in kwargs.items():
                job[key] = value
            
            # Update in database
            self.jobs_container.upsert_item(job)
            
            logger.info(f"Updated job {job_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            # Don't raise here as it's not critical for processing

async def main():
    """Main entry point"""
    processor = AudioCleanerProcessor()
    
    # Start processing messages
    await processor.process_messages()

if __name__ == "__main__":
    asyncio.run(main())
