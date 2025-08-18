"""Centralized Azure client factories for all SDKs (Cosmos, Blob, Service Bus).

Supports either connection strings or Managed Identity (RBAC):
- Cosmos: COSMOS_CONNECTION_STRING or (COSMOS_ACCOUNT_ENDPOINT + USE_MANAGED_IDENTITY=true)
- Blob: AZURE_STORAGE_CONNECTION_STRING or (STORAGE_ACCOUNT_URL + USE_MANAGED_IDENTITY=true)
- Service Bus: AZURE_SERVICE_BUS_CONNECTION_STRING or (SERVICE_BUS_NAMESPACE + USE_MANAGED_IDENTITY=true)
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cosmos globals for caching
_COSMOS_CLIENT = None
_COSMOS_DB = None


def _use_mi() -> bool:
    return os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"


# ---- Cosmos DB clients ----

def _create_cosmos_client():
    """Create a CosmosClient once based on env configuration."""
    global _COSMOS_CLIENT  # noqa: PLW0603
    if _COSMOS_CLIENT is not None:
        return _COSMOS_CLIENT
    try:
        from azure.cosmos import CosmosClient  # type: ignore
    except Exception as e:  # pragma: no cover - import-time guard
        raise RuntimeError(f"azure-cosmos SDK is required: {e}")

    conn = os.getenv("COSMOS_CONNECTION_STRING")
    if conn:
        _COSMOS_CLIENT = CosmosClient.from_connection_string(conn)
        return _COSMOS_CLIENT

    # Optional RBAC route
    endpoint = os.getenv("COSMOS_ACCOUNT_ENDPOINT")
    if endpoint and _use_mi():
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            cred = DefaultAzureCredential()
            _COSMOS_CLIENT = CosmosClient(endpoint, credential=cred)  # type: ignore[arg-type]
            return _COSMOS_CLIENT
        except Exception as e:
            logger.error("Failed to create CosmosClient with DefaultAzureCredential: %s", e)
            raise

    raise RuntimeError("Cosmos configuration missing. Set COSMOS_CONNECTION_STRING or (COSMOS_ACCOUNT_ENDPOINT + USE_MANAGED_IDENTITY=true).")


def get_cosmos_client():
    """Return cached CosmosClient."""
    return _create_cosmos_client()


def get_cosmos_db(name: Optional[str] = None):
    """Return cached database client for configured DB name (default from COSMOS_DB_NAME)."""
    global _COSMOS_DB  # noqa: PLW0603
    if _COSMOS_DB is not None:
        return _COSMOS_DB
    db_name = name or os.getenv("COSMOS_DB_NAME", "AudioCleanerDB")
    cli = get_cosmos_client()
    _COSMOS_DB = cli.get_database_client(db_name)
    return _COSMOS_DB


def get_container(name: str):
    """Shorthand to get a container client from the configured database."""
    db = get_cosmos_db()
    return db.get_container_client(name)


def get_accounts_container():
    return get_container("accounts")


def get_transactions_container():
    return get_container("transactions")


# ---- Blob Storage clients ----


def get_blob_service_client():
    """Return an aio BlobServiceClient using connection string or MI.

    Env options:
      - AZURE_STORAGE_CONNECTION_STRING
      - STORAGE_ACCOUNT_URL (e.g., https://<account>.blob.core.windows.net)
      - USE_MANAGED_IDENTITY=true
    """
    try:
        from azure.storage.blob.aio import BlobServiceClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"azure-storage-blob[aio] SDK is required: {e}")

    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return BlobServiceClient.from_connection_string(conn)

    if _use_mi():
        account_url = os.getenv("STORAGE_ACCOUNT_URL")
        if not account_url:
            raise RuntimeError("STORAGE_ACCOUNT_URL must be set when using managed identity for Blob")
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            cred = DefaultAzureCredential()
            return BlobServiceClient(account_url=account_url, credential=cred)
        except Exception as e:
            logger.error("Failed to create BlobServiceClient with MI: %s", e)
            raise

    raise RuntimeError("Blob configuration missing. Set AZURE_STORAGE_CONNECTION_STRING or (STORAGE_ACCOUNT_URL + USE_MANAGED_IDENTITY=true)")


def get_service_bus_client():
    """Return an aio ServiceBusClient using connection string or MI.

    Env options:
      - AZURE_SERVICE_BUS_CONNECTION_STRING
      - SERVICE_BUS_NAMESPACE (e.g., <ns>.servicebus.windows.net)
      - USE_MANAGED_IDENTITY=true
    """
    try:
        from azure.servicebus.aio import ServiceBusClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"azure-servicebus[aio] SDK is required: {e}")

    conn = os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
    if conn:
        return ServiceBusClient.from_connection_string(conn)

    if _use_mi():
        fqdn = os.getenv("SERVICE_BUS_NAMESPACE")
        if not fqdn:
            raise RuntimeError("SERVICE_BUS_NAMESPACE must be set when using managed identity for Service Bus")
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            cred = DefaultAzureCredential()
            return ServiceBusClient(fully_qualified_namespace=fqdn, credential=cred)
        except Exception as e:
            logger.error("Failed to create ServiceBusClient with MI: %s", e)
            raise

    raise RuntimeError("Service Bus configuration missing. Set AZURE_SERVICE_BUS_CONNECTION_STRING or (SERVICE_BUS_NAMESPACE + USE_MANAGED_IDENTITY=true)")
