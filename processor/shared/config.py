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
    """
    Load strongly validated configuration from environment.
    """

    errors: list[str] = []

    # Core connection strings
    sb = os.getenv('AZURE_SERVICE_BUS_CONNECTION_STRING')
    storage = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    cosmos = os.getenv('COSMOS_CONNECTION_STRING')
    if not sb:
        errors.append('AZURE_SERVICE_BUS_CONNECTION_STRING')
    if not storage:
        errors.append('AZURE_STORAGE_CONNECTION_STRING')
    if not cosmos:
        errors.append('COSMOS_CONNECTION_STRING')

    uploads_container = os.getenv('UPLOADS_CONTAINER_NAME') or ''
    processed_container = os.getenv('PROCESSED_CONTAINER_NAME') or ''
    queue_name = os.getenv('QUEUE_NAME') or ''
    if not uploads_container:
        errors.append('UPLOADS_CONTAINER_NAME')
    if not processed_container:
        errors.append('PROCESSED_CONTAINER_NAME')
    if not queue_name:
        errors.append('QUEUE_NAME')

    # Boolean + numeric parsing
    delete_inputs_raw = os.getenv('DELETE_INPUT_ON_SUCCESS', 'true')
    delete_inputs = delete_inputs_raw.lower() in {'1', 'true', 'yes', 'on'}

    idle_raw = os.getenv('IDLE_SLEEP_SECONDS', '5')
    try:
        idle = float(idle_raw)
    except ValueError:
        logger.warning("Invalid IDLE_SLEEP_SECONDS=%s; defaulting to 5.0", idle_raw)
        idle = 5.0

    if errors:
        raise ValueError(
            "Configuration error (set these env vars for local docker run):\n  - " + "\n  - ".join(errors)
        )

    # Log minimal summary (avoid printing secrets)
    logger.info(
        "Config loaded: containers=(%s,%s) queue=%s idle=%s delete_inputs=%s",
        uploads_container,
        processed_container,
        queue_name,
        idle,
        delete_inputs,
    )

    return Config(
        service_bus_connection=sb,  # type: ignore[arg-type]
        storage_connection=storage,  # type: ignore[arg-type]
        cosmos_connection=cosmos,  # type: ignore[arg-type]
        queue_name=queue_name,  # type: ignore[arg-type]
        uploads_container=uploads_container,
        processed_container=processed_container,
        idle_sleep_seconds=idle,
        delete_inputs_on_success=delete_inputs,
    )
