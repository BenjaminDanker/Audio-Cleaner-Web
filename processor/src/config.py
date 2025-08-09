import os
import logging
from dataclasses import dataclass
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

logger = logging.getLogger(__name__)


def setup_application_insights():
    """Configure Application Insights with code-level opinionated noise reduction.

    Goals:
      * No env var tweaking required for common noise suppression.
      * Disable azure-core instrumentation (removes bulk HTTP dependency spans).
      * Apply a default 5% parent-based trace sampling (can override via env if desired).
      * Provide an internal span filter to drop any remaining low-value Azure noise while keeping custom spans.
    """
    connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
    if not connection_string:
        logger.info("Application Insights not configured (missing APPLICATIONINSIGHTS_CONNECTION_STRING)")
        return
    try:
        # Import only after we set env toggles so auto-configuration reads them.
        from azure.monitor.opentelemetry import configure_azure_monitor  # type: ignore

        # Force-disable dependency auto collection (removes 'dependency' rows); leave logs enabled.
        os.environ['AZURE_MONITOR_AUTO_DEPENDENCIES_ENABLED'] = 'false'

        # Sampler only if user hasn't declared one.
        if not os.getenv('OTEL_TRACES_SAMPLER'):
            provider = TracerProvider(sampler=ParentBased(TraceIdRatioBased(0.05)))
            trace.set_tracer_provider(provider)

        # Instrumentation disable before configure call.
        base_disabled = {
            'azure-core', 'aiohttp-client', 'requests', 'urllib', 'urllib3',
            'psycopg2', 'psycopg'
        }
        existing = os.getenv('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS')
        if existing:
            merged = {x.strip() for x in existing.split(',') if x.strip()}
            merged.update(base_disabled)
            disabled_value = ','.join(sorted(merged))
        else:
            disabled_value = ','.join(sorted(base_disabled))
        os.environ['OTEL_PYTHON_DISABLED_INSTRUMENTATIONS'] = disabled_value

        configure_azure_monitor(connection_string=connection_string, disable_offline_storage=True)

        logger.info("App Insights configured (dependencies disabled, disabled_instr=%s)", disabled_value)
    except Exception as e:
        logger.warning("Failed to configure Application Insights: %s", e)

@dataclass(frozen=True)
class Config:
    service_bus_connection: str
    storage_connection: str
    cosmos_connection: str
    queue_name: str
    uploads_container: str
    processed_container: str
    idle_sleep_seconds: float = 5.0
    delete_inputs_on_success: bool = True


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
    delete_inputs = os.getenv('DELETE_INPUT_ON_SUCCESS', 'true').lower() in {'1','true','yes','on'}
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
        delete_inputs_on_success=delete_inputs,
    )
