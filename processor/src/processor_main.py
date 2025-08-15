import os, json, asyncio, tempfile, logging, traceback, signal, contextlib
from pathlib import Path
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos import CosmosClient
from media_processor import MediaProcessor  # (VideoProcessor alias retained for backward compat)
from config import load_config, setup_application_insights
from job_store import JobStore
from storage_service import BlobStorageService
from ai.asr_pipeline import transcribe_and_translate_file, SubtitleBundle
from captions.caption_encoder import write_srt, write_vtt, Segment as CapSegment
from media_extractor import MediaExtractor
from pricing import extra_language_charge_cents

setup_application_insights()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioCleanerProcessor:
    def __init__(self):
        """Initialize processor dependencies and helpers."""
        # Authentication decision
        self.credential = None
        if os.getenv('USE_MANAGED_IDENTITY', '').lower() == 'true':
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore
                self.credential = DefaultAzureCredential()
                logger.info("Using DefaultAzureCredential for Azure authentication.")
            except Exception as e:
                logger.error(f"Failed to initialize DefaultAzureCredential: {e}")
                raise
        else:
            logger.info("Using connection strings for authentication.")

        # Config
        self.cfg = load_config()

        # Core settings
        self.service_bus_connection = self.cfg.service_bus_connection
        self.queue_name = self.cfg.queue_name
        self.storage_connection = self.cfg.storage_connection
        self.storage_account_name = None

        if not self.storage_connection:
            raise ValueError("Storage connection string is required")

        # Attempt to extract storage account name
        try:
            import re
            match = re.search(r'AccountName=([^;]+)', self.storage_connection)
            if match:
                self.storage_account_name = match.group(1)
        except Exception:
            pass

        # Blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection)
        self.uploads_container = self.cfg.uploads_container
        self.processed_container = self.cfg.processed_container

        # Cosmos DB
        self.cosmos_connection = self.cfg.cosmos_connection
        self.cosmos_client = CosmosClient.from_connection_string(self.cosmos_connection)
        self.database = self.cosmos_client.get_database_client('AudioCleanerDB')
        self.jobs_container = self.database.get_container_client('Jobs')

        # Abstractions
        self.repo = JobStore(self.jobs_container, progress_step=5)
        self.storage = BlobStorageService(self.blob_service_client, self.processed_container)

        # State
        self._stopping = False
        self._install_signal_handlers()
        logger.info("AudioCleanerProcessor initialized")

    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown (best-effort on Windows)."""
        try:
            loop = asyncio.get_event_loop()
            for sig in (getattr(signal, 'SIGTERM', None), getattr(signal, 'SIGINT', None)):
                if sig is None:
                    continue
                try:
                    loop.add_signal_handler(sig, self.request_stop)
                except NotImplementedError:
                    # Likely on Windows where only SIGINT works or in non-main thread
                    pass
        except Exception as e:
            logger.warning(f"Failed to install signal handlers: {e}")

    def request_stop(self):
        if not self._stopping:
            logger.info("Stop requested via signal; will exit after current work item")
            self._stopping = True

    # ---------- Helper utilities ----------

    @staticmethod
    def _parse_message_body(msg):
        body_parts = [part for part in msg.body]
        message_body = b''.join(body_parts).decode('utf-8')
        data = json.loads(message_body)
        if isinstance(data, str):
            data = json.loads(data)
        return data

    async def progress(self, job_id: str, user_id: str, pct: int):
        await self.repo.set_progress(job_id, user_id, pct)
        self._log(event="progress", jobId=job_id, userId=user_id, progress=pct)

    def _log(self, event: str, level: int = logging.INFO, **fields):
        # Simple structured logger; keeps original logger but adds key=value pairs
        kv = ' '.join(f"{k}={fields[k]}" for k in sorted(fields))
        logger.log(level, f"event={event} {kv}")

    async def _is_job_already_final(self, job_id: str, user_id: str) -> bool:
        return await self.repo.is_final(job_id, user_id)

    async def run_continuous(self):
        """Continuously poll the queue and process messages until the platform scales the replica down.
        Designed to be low-idle: waits for messages, sleeps briefly when none are found.
        """
        idle_sleep = self.cfg.idle_sleep_seconds
        logger.info(f"Starting continuous processor loop (idle sleep {idle_sleep}s when queue empty)")
        async with ServiceBusClient.from_connection_string(self.service_bus_connection) as client:
            containers_validated = False
            receiver = client.get_queue_receiver(queue_name=self.queue_name)
            async with receiver:
                while True:
                    if self._stopping:
                        logger.info("Shutdown flag set – exiting run loop after current iteration")
                        break
                    try:
                        if not containers_validated:
                            try:
                                existing = {c['name'] async for c in self.blob_service_client.list_containers()}
                                missing = {self.uploads_container, self.processed_container} - existing
                                if missing:
                                    raise ValueError(f"Missing required storage container(s): {', '.join(missing)}")
                                containers_validated = True
                            except Exception as ce:
                                logger.error(f"Async container validation failed: {ce}")
                                raise
                        msgs = await receiver.receive_messages(max_message_count=1, max_wait_time=15)
                        if not msgs:
                            await asyncio.sleep(idle_sleep)
                            continue
                        msg = msgs[0]
                        try:
                            message_data = self._parse_message_body(msg)
                            logger.info(f"Processing job: {message_data.get('jobId')} (continuous mode)")
                            job_id = message_data.get('jobId')
                            user_id = message_data.get('userId')

                            # Idempotency check
                            if await self._is_job_already_final(job_id, user_id):
                                logger.info(f"Job {job_id} already final state – completing without reprocessing")
                                await receiver.complete_message(msg)
                                continue

                            # Start lock renewal task (in case processing exceeds initial lock)
                            lock_task = asyncio.create_task(self._renew_lock_periodically(receiver, msg))
                            await self.process_video_job(message_data)
                            await receiver.complete_message(msg)
                            logger.info(f"Completed job: {message_data.get('jobId')}")
                            lock_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await lock_task
                        except Exception as e:
                            logger.error(f"Error processing message in continuous loop: {e}")
                            logger.error(traceback.format_exc())
                            if 'message_data' in locals():
                                user_id = message_data.get('userId')
                                await self.update_job_status(
                                    message_data.get('jobId'),
                                    'failed',
                                    user_id,
                                    error_message=str(e)
                                )
                            try:
                                await receiver.dead_letter_message(msg, reason="ProcessingError", error_description=str(e))
                            except Exception as dlq_err:
                                logger.error(f"Failed to dead-letter message: {dlq_err}")
                    except Exception as loop_err:
                        logger.error(f"Top-level continuous loop error: {loop_err}")
                        logger.error(traceback.format_exc())
                        await asyncio.sleep(min(idle_sleep * 2, 30))

    async def _renew_lock_periodically(self, receiver, msg, interval: int = 20):
        """Renew Service Bus message lock periodically until cancelled."""
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await receiver.renew_message_lock(msg)
                    logger.debug("Renewed message lock")
                except Exception as e:
                    logger.warning(f"Lock renewal failed (may be completed or timed out): {e}")
                    break
        except asyncio.CancelledError:
            pass

    async def process_video_job(self, job_data):
        """Process a single video job"""
        job_id = job_data['jobId']
        user_id = job_data['userId']

        try:
            record = await self.repo.get(job_id, user_id)
        except Exception as e:
            self._log(level=logging.ERROR, event="job_fetch_failed", jobId=job_id, userId=user_id, error=str(e))
            raise ValueError(f"Job record not found: {job_id}")
        file_ext = Path(record.file_name).suffix or '.mp4'
        output_file_name = f"{record.user_id}/{record.id}_processed{file_ext}"
        atten_db = record.attenuation_db
        processing_type = record.processing_type
        input_blob_url = record.input_blob_url

        self._log(event="job_start", jobId=job_id, userId=user_id, output=output_file_name, type=processing_type)

        # Update job status to processing
        await self.progress(job_id, user_id, 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download input video from blob storage using the actual URL
                await self.progress(job_id, user_id, 15)
                input_file_path = await self.storage.download(input_blob_url, temp_dir, user_id, job_id)
                await self.progress(job_id, user_id, 25)

                # Create a mock uploaded file object for VideoProcessor
                # Extract audio
                await self.progress(job_id, user_id, 30)
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

                self._log(event="processing_begin", jobId=job_id)

                # Process video using MediaProcessor with real progress callback
                await self.progress(job_id, user_id, 35)
                processor = MediaProcessor(mock_file, atten_db, processing_type=processing_type or "denoise")

                try:
                    self._log(event="invoke_processor", jobId=job_id)

                    progress_milestones = {10:45, 30:60, 85:80}  # task-level -> job progress mapping

                    # Thread-safe bridge for task progress (task runs in executor)
                    def task_progress_bridge(pct: int):  # called from loop thread via call_soon_threadsafe
                        mapped = None
                        # Pick the highest milestone <= pct
                        for k in sorted(progress_milestones):
                            if pct >= k:
                                mapped = progress_milestones[k]
                        if mapped is not None:
                            asyncio.run_coroutine_threadsafe(self.progress(job_id, user_id, mapped), asyncio.get_event_loop())

                    async def run_processing():
                        loop = asyncio.get_event_loop()
                        # Provide a wrapper that schedules the bridge onto loop thread
                        def progress_cb(p: int):
                            loop.call_soon_threadsafe(task_progress_bridge, p)
                        return await loop.run_in_executor(None, lambda: processor.process(progress_cb))

                    output_path = await run_processing()

                    self._log(event="processing_done", jobId=job_id, output=output_path)

                    # Upload processed video to blob storage
                    self._log(event="upload_begin", jobId=job_id)
                    await self.progress(job_id, user_id, 85)
                    download_url = await self.storage.upload_processed(output_path, output_file_name)
                    self._log(event="upload_done", jobId=job_id, url=download_url)

                    await self.progress(job_id, user_id, 95)

                    # Optional: Transcription + translations -> SRT/VTT
                    subtitles_urls = {}
                    try:
                        # Look for requested languages in job record metadata
                        requested_langs = []
                        try:
                            requested_langs = record.metadata.get('languagesRequested', []) or []
                        except Exception:
                            requested_langs = []
                        if requested_langs:
                            # Extract WAV from processed output for ASR
                            extractor = MediaExtractor(16000)
                            extraction = extractor.extract(output_path, temp_dir)
                            wav_for_asr = extraction.extracted_wav_path
                            bundle: SubtitleBundle = transcribe_and_translate_file(wav_for_asr, 16000, requested_langs)
                            # Persist SRT/VTT per language
                            for lang, segs in bundle.segments_by_lang.items():
                                cap_segments = [CapSegment(s.start, s.end, s.text) for s in segs]
                                base_name = f"{record.user_id}/{record.id}_{lang}"
                                srt_path = os.path.join(temp_dir, f"{lang}.srt")
                                vtt_path = os.path.join(temp_dir, f"{lang}.vtt")
                                write_srt(cap_segments, srt_path)
                                write_vtt(cap_segments, vtt_path)
                                srt_blob = f"{base_name}.srt"
                                vtt_blob = f"{base_name}.vtt"
                                srt_url = await self.storage.upload_processed(srt_path, srt_blob)
                                vtt_url = await self.storage.upload_processed(vtt_path, vtt_blob)
                                subtitles_urls[lang] = {"srt": srt_url, "vtt": vtt_url}
                    except Exception as sub_err:
                        self._log(level=logging.WARNING, event="subtitles_failed", jobId=job_id, error=str(sub_err))

                    # Update job status to completed
                    await self.update_job_status(
                        job_id,
                        'completed',
                        user_id,
                        progress=100,
                        downloadUrl=download_url,
                        outputBlobName=output_file_name,
                        subtitles=subtitles_urls
                    )

                    # Post-completion: apply extra language credit deduction if any
                    try:
                        additional_langs = 0
                        try:
                            requested_langs = record.metadata.get('languagesRequested', []) or []
                            # Count languages beyond the primary
                            additional_langs = max(0, len(requested_langs) - 1)
                        except Exception:
                            additional_langs = 0
                        if additional_langs > 0:
                            # Approx minutes from extracted WAV duration
                            try:
                                import soundfile as sf
                                import numpy as np
                                # Extract WAV from the processed output to measure duration
                                extractor2 = MediaExtractor(16000)
                                extraction2 = extractor2.extract(output_path, temp_dir)
                                wav_path2 = extraction2.extracted_wav_path
                                data, rate = sf.read(wav_path2, dtype='float32')
                                samples = data.shape[0] if getattr(data, 'shape', None) else len(data)
                                dur_min = float(samples) / float(rate) / 60.0
                            except Exception:
                                dur_min = 0.0
                            extra_cents = extra_language_charge_cents(dur_min, additional_langs)
                            if extra_cents > 0:
                                # Deduct from accounts container and create a transaction record
                                accounts = self.database.get_container_client('accounts')
                                txns = self.database.get_container_client('transactions')
                                # Fetch account
                                acc = accounts.read_item(user_id, user_id)
                                bal = int(acc.get('balance', 0))
                                if bal >= extra_cents:
                                    acc['balance'] = bal - extra_cents
                                    acc['updatedAt'] = self.repo._now_iso()
                                    accounts.upsert_item(acc)
                                    tid = f"txn_lang_{job_id}"
                                    tx = {
                                        'id': tid,
                                        'userId': user_id,
                                        'type': 'translation-extra',
                                        'amount': extra_cents,
                                        'description': f'Extra languages ({additional_langs}) for job {job_id}',
                                        'jobId': job_id,
                                        'createdAt': self.repo._now_iso(),
                                    }
                                    txns.upsert_item(tx)
                                    self._log(event="extra_lang_deducted", jobId=job_id, userId=user_id, cents=extra_cents)
                                else:
                                    self._log(level=logging.WARNING, event="extra_lang_insufficient_balance", jobId=job_id, needed=extra_cents, balance=bal)
                    except Exception as credit_err:
                        self._log(level=logging.WARNING, event="extra_lang_credit_error", jobId=job_id, error=str(credit_err))

                    # Delete the input blob now that processing is complete (optional)
                    if self.cfg.delete_inputs_on_success:
                        try:
                            logger.info(f"Attempting to delete input blob for job {job_id}: {input_blob_url}")
                            # Simple transient retry (e.g. if lease/replication delay) up to 3 times
                            last_err = None
                            for attempt in range(1,4):
                                try:
                                    await self.storage.delete(input_blob_url)
                                    logger.info(f"Successfully deleted input blob for job {job_id} attempt={attempt}")
                                    last_err = None
                                    break
                                except Exception as inner_del_err:
                                    last_err = inner_del_err
                                    logger.warning(f"Delete attempt {attempt} failed for job {job_id}: {inner_del_err}")
                                    await asyncio.sleep(1 * attempt)
                            if last_err:
                                raise last_err
                        except Exception as delete_error:
                            logger.error(f"Failed to delete input blob for job {job_id}: {delete_error}")
                            logger.error(f"Blob URL was: {input_blob_url}")
                            # Don't fail the job if we can't delete the input blob, but log it for investigation
                    else:
                        logger.info(f"Configured to retain input blob for job {job_id}: {input_blob_url}")

                    self._log(event="job_complete", jobId=job_id)

                finally:
                    # Cleanup processor temp files
                    processor.immediate_cleanup(logger)

            except Exception as e:
                self._log(level=logging.ERROR, event="job_error", jobId=job_id, error=str(e), trace=traceback.format_exc())
                await self.update_job_status(job_id, 'failed', user_id, error_message=str(e))
                raise

    async def update_job_status(self, job_id, status, user_id=None, **kwargs):
        await self.repo.update_status(job_id, status, user_id, **kwargs)

async def main():
    """Entry point: continuous consumption only (minimal code)."""
    processor = AudioCleanerProcessor()
    await processor.run_continuous()
    try:
        await processor.blob_service_client.close()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        processor.cosmos_client.close()

if __name__ == "__main__":
    asyncio.run(main())
