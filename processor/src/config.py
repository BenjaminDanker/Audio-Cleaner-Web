import os
import logging
from dataclasses import dataclass
from azure.monitor.opentelemetry import configure_azure_monitor

logger = logging.getLogger(__name__)

def setup_application_insights():
    connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
    if not connection_string:
        logger.info("Application Insights not configured (missing APPLICATIONINSIGHTS_CONNECTION_STRING)")
        return
    try:
        os.environ['OTEL_PYTHON_DISABLED_INSTRUMENTATIONS'] = 'psycopg2,psycopg'
        configure_azure_monitor(connection_string=connection_string, disable_offline_storage=True)
        logger.info("Application Insights configured")
    except Exception as e:
        logger.warning(f"Failed to configure Application Insights: {e}")

@dataclass(frozen=True)
class Config:
    service_bus_connection: str
    storage_connection: str
    cosmos_connection: str
    queue_name: str
    uploads_container: str
    processed_container: str
    idle_sleep_seconds: float = 5.0


def load_config() -> Config:
    sb = os.getenv('AZURE_SERVICE_BUS_CONNECTION_STRING')
    if not sb:
        raise ValueError("AZURE_SERVICE_BUS_CONNECTION_STRING required")
    storage = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not storage:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING required")
    cosmos = os.getenv('COSMOS_CONNECTION_STRING')
    if not cosmos:
        raise ValueError("COSMOS_CONNECTION_STRING required")
    idle = float(os.getenv('IDLE_SLEEP_SECONDS', '5'))
    uploads_container = os.getenv('UPLOADS_CONTAINER_NAME')
    processed_container = os.getenv('PROCESSED_CONTAINER_NAME')
    queue_name = os.getenv('QUEUE_NAME') or os.getenv('AZURE_SERVICE_BUS_QUEUE_NAME')
    missing = [name for name, val in [
        ("UPLOADS_CONTAINER_NAME", uploads_container),
        ("PROCESSED_CONTAINER_NAME", processed_container),
        ("QUEUE_NAME", queue_name)
    ] if not val]
    if missing:
        raise ValueError("Missing required environment variable(s): " + ", ".join(missing))
    return Config(
        service_bus_connection=sb,
        storage_connection=storage,
        cosmos_connection=cosmos,
        queue_name=queue_name,
        uploads_container=uploads_container,
        processed_container=processed_container,
        idle_sleep_seconds=idle,
    )
