# Output Values
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.main.name
}

output "storage_connection_string" {
  description = "Storage account connection string"
  value       = azurerm_storage_account.main.primary_connection_string
  sensitive   = true
}

output "servicebus_connection_string" {
  description = "Service Bus connection string"
  value       = azurerm_servicebus_namespace.main.default_primary_connection_string
  sensitive   = true
}

output "cosmos_connection_string" {
  description = "Cosmos DB connection string"
  value       = azurerm_cosmosdb_account.main.primary_sql_connection_string
  sensitive   = true
}

output "container_app_environment_id" {
  description = "Container App Environment ID"
  value       = azurerm_container_app_environment.main.id
}

output "managed_identity_id" {
  description = "User Assigned Managed Identity ID"
  value       = azurerm_user_assigned_identity.processor.id
}

output "managed_identity_client_id" {
  description = "User Assigned Managed Identity Client ID"
  value       = azurerm_user_assigned_identity.processor.client_id
}

output "static_web_app_url" {
  description = "URL of the Static Web App"
  value       = "https://${azurerm_static_web_app.main.default_host_name}"
}

output "static_web_app_api_key" {
  description = "API key for deploying to Static Web App"
  value       = azurerm_static_web_app.main.api_key
  sensitive   = true
}

output "application_insights_instrumentation_key" {
  description = "Application Insights Instrumentation Key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

output "application_insights_connection_string" {
  description = "Application Insights Connection String"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "speech_services_endpoint" {
  description = "Azure AI Speech Services endpoint"
  value       = azurerm_cognitive_account.speech_services.endpoint
}

output "speech_services_region" {
  description = "Azure AI Speech Services region"
  value       = azurerm_cognitive_account.speech_services.location
}

output "openai_endpoint" {
  description = "Azure OpenAI endpoint (for cleanup only)"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "translator_endpoint" {
  description = "Azure Translator endpoint"
  value       = azurerm_cognitive_account.translator.endpoint
}
