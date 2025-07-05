"""
Local Development Processor Service
Handles video processing for local development without requiring full Azure infrastructure
Includes Flask web interface for health checks and status monitoring
"""

import os
import json
import time
import asyncio
import tempfile
import logging
import shutil
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify
from video_handler import VideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app for health checks and coordination
app = Flask(__name__)

class LocalProcessor:
    def __init__(self):
        self.jobs_dir = Path("api/temp/jobs")
        self.uploads_dir = Path("api/temp/uploads")
        self.downloads_dir = Path("api/temp/downloads")
        
        # Ensure directories exist
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        self.running = True
        logger.info("LocalProcessor initialized")

    def get_job_status_file(self, job_id):
        return self.jobs_dir / f"{job_id}.json"

    def update_job_status(self, job_id, status, progress=None, message=None, download_url=None):
        """Update job status in local file"""
        try:
            status_file = self.get_job_status_file(job_id)
            
            # Read existing data if file exists
            if status_file.exists():
                with open(status_file, 'r') as f:
                    job_data = json.load(f)
            else:
                job_data = {
                    'id': job_id,
                    'fileName': 'unknown.mp4',
                    'processingType': 'denoise',
                    'createdAt': datetime.now().isoformat(),
                    'status': 'queued',
                    'progress': 0
                }
            
            # Update fields
            job_data['status'] = status
            job_data['updatedAt'] = datetime.now().isoformat()
            
            if progress is not None:
                job_data['progress'] = progress
            if message is not None:
                job_data['message'] = message
            if download_url is not None:
                job_data['downloadUrl'] = download_url
            if status == 'completed':
                job_data['completedAt'] = datetime.now().isoformat()
                job_data['progress'] = 100
            
            # Write updated data
            with open(status_file, 'w') as f:
                json.dump(job_data, f, indent=2)
                
            logger.info(f"Updated job {job_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")

    def process_video_file(self, job_id, file_path, atten_db=30):
        """Process a video file with denoising"""
        try:
            self.update_job_status(job_id, 'processing', 10, 'Starting video processing...')
            
            # Create a simple file upload mock
            class SimpleFileUpload:
                def __init__(self, file_path):
                    self.file_path = Path(file_path)
                    self.filename = self.file_path.name
                
                def save(self, target_path):
                    shutil.copy2(self.file_path, target_path)
            
            mock_upload = SimpleFileUpload(file_path)
            
            # Use the VideoProcessor to process the file
            processor = VideoProcessor()
            
            self.update_job_status(job_id, 'processing', 30, 'Processing video...')
            
            result = processor.process_video(mock_upload, atten_db=atten_db)
            
            if result['success']:
                # Move the processed file to downloads directory
                processed_file = Path(result['output_path'])
                download_filename = f"processed_{job_id}_{processed_file.name}"
                download_path = self.downloads_dir / download_filename
                
                shutil.move(processed_file, download_path)
                
                # Create download URL (for local development)
                download_url = f"local://downloads/{download_filename}"
                
                self.update_job_status(
                    job_id, 
                    'completed', 
                    100, 
                    'Video processing completed successfully',
                    download_url
                )
                
                logger.info(f"Job {job_id} completed successfully")
                
            else:
                self.update_job_status(
                    job_id, 
                    'failed', 
                    0, 
                    f'Processing failed: {result.get("error", "Unknown error")}'
                )
                
        except Exception as e:
            logger.error(f"Error processing video for job {job_id}: {e}")
            self.update_job_status(
                job_id, 
                'failed', 
                0, 
                f'Processing error: {str(e)}'
            )

    async def process_pending_jobs(self):
        """Check for and process pending jobs"""
        if not self.running:
            return
            
        try:
            # Look for queued jobs
            for status_file in self.jobs_dir.glob("*.json"):
                try:
                    with open(status_file, 'r') as f:
                        job_data = json.load(f)
                    
                    if job_data.get('status') == 'queued':
                        job_id = job_data['id']
                        logger.info(f"Found queued job: {job_id}")
                        
                        # Get the file path from job data, fallback to test video
                        file_path = job_data.get('filePath', 'video/C1395.MP4')
                        
                        if Path(file_path).exists():
                            # Process in background thread to not block the loop
                            thread = threading.Thread(
                                target=self.process_video_file,
                                args=(job_id, file_path)
                            )
                            thread.daemon = True
                            thread.start()
                        else:
                            self.update_job_status(
                                job_id, 
                                'failed', 
                                0, 
                                f'Input file not found: {file_path}'
                            )
                
                except Exception as e:
                    logger.error(f"Error processing job file {status_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")

    async def run_processor_loop(self):
        """Main processing loop"""
        logger.info("Starting local processor loop...")
        
        while self.running:
            try:
                await self.process_pending_jobs()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the processor"""
        self.running = False
        logger.info("Processor stopped")

# Global processor instance
processor = None

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'local-processor',
        'timestamp': datetime.now().isoformat(),
        'jobs_dir': str(Path("api/temp/jobs").resolve()),
        'running': processor.running if processor else False
    })

@app.route('/jobs')
def list_jobs():
    """List all jobs and their statuses"""
    try:
        jobs_dir = Path("api/temp/jobs")
        jobs = []
        
        for status_file in jobs_dir.glob("*.json"):
            try:
                with open(status_file, 'r') as f:
                    job_data = json.load(f)
                    jobs.append(job_data)
            except Exception as e:
                logger.error(f"Error reading job file {status_file}: {e}")
        
        return jsonify({
            'jobs': jobs,
            'total': len(jobs),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/<job_id>/status')
def get_job_status(job_id):
    """Get status of a specific job"""
    try:
        status_file = Path("api/temp/jobs") / f"{job_id}.json"
        
        if not status_file.exists():
            return jsonify({'error': 'Job not found'}), 404
        
        with open(status_file, 'r') as f:
            job_data = json.load(f)
        
        return jsonify(job_data)
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500

def run_processor():
    """Run the processor in a separate thread"""
    global processor
    processor = LocalProcessor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(processor.run_processor_loop())

def start_background_processor():
    """Start the background processor thread"""
    processor_thread = threading.Thread(target=run_processor, daemon=True)
    processor_thread.start()
    logger.info("Background processor started")

if __name__ == '__main__':
    print("🚀 Starting Local Audio Cleaner Processor...")
    print("📊 Health check: http://localhost:8080/health")
    print("📋 Jobs status: http://localhost:8080/jobs")
    print()
    
    # Start the background processor
    start_background_processor()
    
    # Start the Flask app for health checks and status
    app.run(host='0.0.0.0', port=8080, debug=False)
