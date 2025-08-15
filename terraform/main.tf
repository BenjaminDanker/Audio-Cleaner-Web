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
  name                  = var.uploads_container_name
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "output" {
  name                  = var.processed_container_name
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
  prefix_match = ["${var.uploads_container_name}/"]
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
  prefix_match = ["${var.processed_container_name}/"]
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
  name         = var.queue_name
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

# Cosmos container for API Keys (hashed storage)
resource "azurerm_cosmosdb_sql_container" "api_keys" {
  name                = var.cosmos_api_keys_container_name
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_sql_database.main.name

  partition_key_paths = ["/id"]
}

# Application Insights
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  # Retention: Azure Log Analytics minimum is 30 days for PerGB2018; cannot go lower (7 or 15 not supported).
  retention_in_days   = 30

  tags = var.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  daily_data_cap_in_gb                  = 2
  daily_data_cap_notifications_disabled = true

  tags = var.tags
}

# Azure OpenAI account (Cognitive Services - OpenAI)
resource "azurerm_cognitive_account" "openai" {
  name                = "aoai-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"
  public_network_access_enabled = true

  tags = var.tags
}

data "azurerm_cognitive_account_api_keys" "openai" {
  resource_group_name = azurerm_resource_group.main.name
  name                = azurerm_cognitive_account.openai.name
}

# Optional: model deployments for Whisper and Chat cleanup
resource "azurerm_cognitive_deployment" "openai_whisper" {
  count                = length(var.openai_whisper_deployment) > 0 && length(var.openai_whisper_model_name) > 0 ? 1 : 0
  name                 = var.openai_whisper_deployment
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.openai_whisper_model_name
    version = var.openai_whisper_model_version
  }
  sku {
    name     = "GlobalStandard"
    capacity = 1
  }
}

resource "azurerm_cognitive_deployment" "openai_chat" {
  count                = length(var.openai_chat_deployment) > 0 && length(var.openai_chat_model_name) > 0 ? 1 : 0
  name                 = var.openai_chat_deployment
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.openai_chat_model_name
    version = var.openai_chat_model_version
  }
  sku {
    name     = "GlobalStandard"
    capacity = 1
  }
}

# Azure Translator (TextTranslation)
resource "azurerm_cognitive_account" "translator" {
  name                = "tr-${local.project_name}-${local.resource_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "TextTranslation"
  sku_name            = "S1"
  public_network_access_enabled = true

  tags = var.tags
}

data "azurerm_cognitive_account_api_keys" "translator" {
  resource_group_name = azurerm_resource_group.main.name
  name                = azurerm_cognitive_account.translator.name
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
  "UPLOADS_CONTAINER_NAME"                = var.uploads_container_name
  "PROCESSED_CONTAINER_NAME"              = var.processed_container_name
  "QUEUE_NAME"                            = var.queue_name
  # API Keys + Cosmos names for Functions/APIs
  "STREAMING_API_KEYS"                    = var.streaming_api_keys
  "COSMOS_DB_NAME"                        = azurerm_cosmosdb_sql_database.main.name
  "COSMOS_API_KEYS_CONTAINER"             = var.cosmos_api_keys_container_name
  # Azure OpenAI + Translator (for Functions if needed later)
  # Azure OpenAI / Translator for Functions (if needed)
  "AZURE_OPENAI_ENDPOINT"                 = length(var.openai_endpoint) > 0 ? var.openai_endpoint : azurerm_cognitive_account.openai.endpoint
  "AZURE_OPENAI_API_VERSION"              = var.openai_api_version
  "AZURE_OPENAI_WHISPER_DEPLOYMENT"       = var.openai_whisper_deployment
  "AZURE_OPENAI_CLEANUP_DEPLOYMENT"       = var.openai_chat_deployment
  "AZURE_TRANSLATOR_REGION"               = var.translator_region
  "AZURE_TRANSLATOR_ENDPOINT"             = "https://api.cognitive.microsofttranslator.com"
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

      # Azure OpenAI & Translator configuration for file/batch pipeline
      # Azure OpenAI env expected by processor code
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = length(var.openai_endpoint) > 0 ? var.openai_endpoint : azurerm_cognitive_account.openai.endpoint
      }
      env {
        name  = "AZURE_OPENAI_API_VERSION"
        value = var.openai_api_version
      }
      env {
        name  = "AZURE_OPENAI_WHISPER_DEPLOYMENT"
        value = var.openai_whisper_deployment
      }
      env {
        name  = "AZURE_OPENAI_CLEANUP_DEPLOYMENT"
        value = var.openai_chat_deployment
      }
      # Translator env expected by processor code
      env {
        name  = "AZURE_TRANSLATOR_REGION"
        value = var.translator_region
      }
      env {
        name  = "AZURE_TRANSLATOR_ENDPOINT"
        value = "https://api.cognitive.microsofttranslator.com"
      }
      env {
        name  = "STREAMING_API_KEYS"
        value = var.streaming_api_keys
      }
      env {
        name  = "COSMOS_DB_NAME"
        value = azurerm_cosmosdb_sql_database.main.name
      }
      env {
        name  = "COSMOS_API_KEYS_CONTAINER"
        value = var.cosmos_api_keys_container_name
      }

      env {
        name        = "AZURE_OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }
      env {
        name        = "AZURE_TRANSLATOR_KEY"
        secret_name = "translator-key"
      }

      # Propagate container names for future code configurability
      env {
        name  = "UPLOADS_CONTAINER_NAME"
        value = var.uploads_container_name
      }
      env {
        name  = "PROCESSED_CONTAINER_NAME"
        value = var.processed_container_name
      }

      env {
        name  = "QUEUE_NAME"
        value = var.queue_name
      }

      env {
        name  = "REFUND_API_ENDPOINT"
        value = "https://${azurerm_static_web_app.main.default_host_name}/api/refund-failed-job"
      }
    }

    # Service Bus queue scaling rule
    custom_scale_rule {
      name             = "servicebus-queue-scale"
      custom_rule_type = "azure-servicebus"
      metadata = {
        queueName    = var.queue_name
        messageCount = "1"  # scale out as soon as 1 message pending
      }
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

  # Secrets for external APIs
  secret {
    name  = "openai-api-key"
    value = (length(var.openai_api_key) > 0 ? var.openai_api_key : data.azurerm_cognitive_account_api_keys.openai.primary_key)
  }
  secret {
    name  = "translator-key"
    value = (length(var.translator_key) > 0 ? var.translator_key : data.azurerm_cognitive_account_api_keys.translator.primary_key)
  }

  tags = var.tags
}

# Streaming Container App (FastAPI WS service)
resource "azurerm_container_app" "streaming" {
  name                         = "ca-stream-${local.resource_suffix}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.processor.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.processor.id
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.streaming_min_replicas
    max_replicas = var.streaming_max_replicas

    container {
      name   = "streaming"
      image  = "${azurerm_container_registry.main.login_server}/${var.streaming_image_name}:${var.streaming_image_tag}"
      cpu    = 1.0
      memory = "1.5Gi"

      env {
        name  = "STREAMING_API_KEYS"
        value = var.streaming_api_keys
      }
      # Azure OpenAI env expected by streaming code
      env {
        name  = "AZURE_OPENAI_ENDPOINT"
        value = length(var.openai_endpoint) > 0 ? var.openai_endpoint : azurerm_cognitive_account.openai.endpoint
      }
      env {
        name  = "AZURE_OPENAI_API_VERSION"
        value = var.openai_api_version
      }
      env {
        name  = "AZURE_OPENAI_WHISPER_DEPLOYMENT"
        value = var.openai_whisper_deployment
      }
      env {
        name  = "AZURE_OPENAI_CLEANUP_DEPLOYMENT"
        value = var.openai_chat_deployment
      }
      env {
        name        = "AZURE_OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }
      env {
        name  = "AZURE_TRANSLATOR_REGION"
        value = var.translator_region
      }
      env {
        name        = "AZURE_TRANSLATOR_KEY"
        secret_name = "translator-key"
      }
      env {
        name  = "AZURE_TRANSLATOR_ENDPOINT"
        value = "https://api.cognitive.microsofttranslator.com"
      }
      env {
        name        = "COSMOS_CONNECTION_STRING"
        secret_name = "cosmos-connectionstring"
      }
      env {
        name  = "AZURE_STORAGE_CONNECTION_STRING"
        value = azurerm_storage_account.main.primary_connection_string
      }
      env {
        name  = "PROCESSED_CONTAINER_NAME"
        value = var.processed_container_name
      }
      env {
        name  = "COSMOS_DB_NAME"
        value = azurerm_cosmosdb_sql_database.main.name
      }
      env {
        name  = "COSMOS_API_KEYS_CONTAINER"
        value = var.cosmos_api_keys_container_name
      }
    }
  }

  secret {
    name  = "openai-api-key"
    value = var.openai_api_key
  }
  secret {
    name  = "translator-key"
    value = var.translator_key
  }
  secret {
    name  = "cosmos-connectionstring"
    value = azurerm_cosmosdb_account.main.primary_sql_connection_string
  }

  tags = var.tags
}
