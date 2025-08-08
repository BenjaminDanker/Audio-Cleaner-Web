# Audio Cleaner Pro - Terraform Infrastructure
# Target: Static Web Apps + Azure Functions + Container Apps for Processor

# Generate a random suffix for unique naming
resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  resource_suffix = random_id.suffix.hex
  location       = var.location
  project_name   = "audioclean"  # Shortened to avoid length issues
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.project_name}-${local.resource_suffix}"
  location = local.location

  tags = var.tags
}

# Storage Account for blob storage
resource "azurerm_storage_account" "main" {
  name                     = "st${replace(local.project_name, "-", "")}${local.resource_suffix}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version         = "TLS1_2"

  blob_properties {
    cors_rule {
      allowed_headers    = ["*"]
      allowed_methods    = ["DELETE", "GET", "HEAD", "MERGE", "POST", "OPTIONS", "PUT"]
      allowed_origins    = ["*"]
      exposed_headers    = ["*"]
      max_age_in_seconds = 3600
    }
  }

  tags = var.tags
}

# Storage containers
resource "azurerm_storage_container" "input" {
  name                  = "uploads"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "output" {
  name                  = "processed-videos"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

# Lifecycle management policy for automatic tier transitions
resource "azurerm_storage_management_policy" "main" {
  storage_account_id = azurerm_storage_account.main.id

  rule {
    name    = "upload-files-hot-to-delete"
    enabled = true

    filters {
      prefix_match = ["uploads/"]
      blob_types   = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_creation_greater_than = 1
      }
    }
  }

  rule {
    name    = "processed-videos-cleanup"
    enabled = true

    filters {
      prefix_match = ["processed-videos/"]
      blob_types   = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_creation_greater_than = 3
      }
    }
  }

  rule {
    name    = "general-cleanup"
    enabled = true

    filters {
      blob_types = ["blockBlob"]
    }

    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than = 30
        tier_to_cold_after_days_since_modification_greater_than = 90
      }
    }
  }
}

# Service Bus for job queue
resource "azurerm_servicebus_namespace" "main" {
  name                = "sb-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Basic"

  tags = var.tags
}

resource "azurerm_servicebus_queue" "video_jobs" {
  name         = "video-processing-jobs"
  namespace_id = azurerm_servicebus_namespace.main.id
  max_delivery_count                  = 10
  default_message_ttl                 = "PT1H"
  dead_lettering_on_message_expiration = true
}

# Cosmos DB for job metadata
resource "azurerm_cosmosdb_account" "main" {
  name                = "cosmos-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"
  free_tier_enabled   = true

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  tags = var.tags
}

resource "azurerm_cosmosdb_sql_database" "main" {
  name                = "AudioCleanerDB"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  throughput = 1000
}

resource "azurerm_cosmosdb_sql_container" "jobs" {
  name                = "Jobs"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  
  partition_key_paths = ["/userId"]
}

resource "azurerm_cosmosdb_sql_container" "accounts" {
  name                = "accounts"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  
  partition_key_paths = ["/userId"]
}

resource "azurerm_cosmosdb_sql_container" "transactions" {
  name                = "transactions"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name
  
  partition_key_paths = ["/userId"]
}

# Application Insights
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  tags = var.tags
}

# Static Web App (with integrated Azure Functions)
resource "azurerm_static_web_app" "main" {
  name                = "swa-${local.project_name}-${local.resource_suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = "Central US"  # Updated to Central US
  sku_tier            = "Standard"
  sku_size            = "Standard"

  identity {
    type = "SystemAssigned"
  }

  app_settings = {
    "AZURE_STORAGE_CONNECTION_STRING"       = azurerm_storage_account.main.primary_connection_string
    "AZURE_SERVICE_BUS_CONNECTION_STRING"   = azurerm_servicebus_namespace.main.default_primary_connection_string
    "COSMOS_CONNECTION_STRING"              = azurerm_cosmosdb_account.main.primary_sql_connection_string
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
    "STRIPE_SECRET_KEY"                     = var.stripe_secret_key
    "STRIPE_PUBLIC_KEY"                     = var.stripe_public_key
    "STRIPE_WEBHOOK_SECRET"                 = var.stripe_webhook_secret
    "FRONTEND_URL"                          = var.frontend_url
    "STRIPE_TOPUP_PRICE_ID"                 = var.stripe_topup_price_id
  }

  tags = var.tags
}

# Get current client configuration
data "azurerm_client_config" "current" {}

// Azure Container Registry for storing processor image
resource "azurerm_container_registry" "main" {
  name                     = "acr${local.project_name}${local.resource_suffix}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  sku                      = "Basic"
  admin_enabled            = true

  tags = var.tags
}

# Container Apps Environment for Python Processor
resource "azurerm_container_app_environment" "main" {
  name                = "cae-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  tags = var.tags
}

# User-assigned managed identity for Container Apps
resource "azurerm_user_assigned_identity" "processor" {
  name                = "id-${local.project_name}-processor-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = var.tags
}

# RBAC assignments for the managed identity
resource "azurerm_role_assignment" "storage_blob_contributor" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.processor.principal_id
}

resource "azurerm_role_assignment" "servicebus_data_receiver" {
  scope                = azurerm_servicebus_namespace.main.id
  role_definition_name = "Azure Service Bus Data Receiver"
  principal_id         = azurerm_user_assigned_identity.processor.principal_id
}

resource "azurerm_role_assignment" "cosmos_contributor" {
  scope                = azurerm_cosmosdb_account.main.id
  role_definition_name = "DocumentDB Account Contributor"
  principal_id         = azurerm_user_assigned_identity.processor.principal_id
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.processor.principal_id
}

# Container App for Python Processor
resource "azurerm_container_app" "processor" {
  name                         = "ca-proc-${local.resource_suffix}"  # Shortened to fit 32 char limit
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.processor.id]
  }

  # Registry configuration for ACR authentication
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.processor.id
  }

  template {
    min_replicas = 0
    max_replicas = 10

    container {
      name   = "processor"
      image  = "${azurerm_container_registry.main.login_server}/audio-cleaner-processor:latest"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.processor.client_id
      }

      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }

      env {
        name  = "USE_MANAGED_IDENTITY"
        value = "true"
      }

      env {
        name  = "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"
        value = "psycopg2,psycopg"
      }

      env {
        name        = "AZURE_SERVICE_BUS_CONNECTION_STRING"
        secret_name = "servicebus-connectionstring"
      }

      env {
        name        = "AZURE_STORAGE_CONNECTION_STRING"
        secret_name = "storage-connectionstring"
      }

      env {
        name        = "COSMOS_CONNECTION_STRING"
        secret_name = "cosmos-connectionstring"
      }

      env {
        name  = "REFUND_API_ENDPOINT"
        value = "https://${azurerm_static_web_app.main.default_host_name}/api/refund-failed-job"
      }
    }

    # KEDA Service Bus scaling rule
    azure_queue_scale_rule {
      name         = "servicebus-scale-rule"
      queue_name   = "video-processing-jobs"
      queue_length = 1

      authentication {
        secret_name       = "servicebus-connectionstring"
        trigger_parameter = "connection"
      }
    }
  }

  # Service Bus connection string secret for KEDA
  secret {
    name  = "servicebus-connectionstring"
    value = azurerm_servicebus_namespace.main.default_primary_connection_string
  }

  # Storage connection string for processor
  secret {
    name  = "storage-connectionstring"
    value = azurerm_storage_account.main.primary_connection_string
  }

  # Cosmos connection string for processor
  secret {
    name  = "cosmos-connectionstring"
    value = azurerm_cosmosdb_account.main.primary_sql_connection_string
  }

  tags = var.tags
}
