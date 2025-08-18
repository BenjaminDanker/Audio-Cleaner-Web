# Input Variables
variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "audio-cleaner"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region where resources will be deployed"
  type        = string
  default     = "Central US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "AudioCleanerPro"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}

# Stripe Configuration
variable "stripe_secret_key" {
  description = "Stripe secret key"
  type        = string
  sensitive   = true
}

variable "stripe_public_key" {
  description = "Stripe public key"
  type        = string
}

variable "stripe_webhook_secret" {
  description = "Stripe webhook secret"
  type        = string
  sensitive   = true
}

# Stripe top-up price (one-time). Keeping non-sensitive (price IDs are public-safe)
variable "stripe_topup_price_id" {
  description = "Stripe Price ID used for account top-ups (one-time payment)"
  type        = string
  default     = ""
}

variable "frontend_url" {
  description = "Frontend URL for Stripe redirects"
  type        = string
  default     = ""
}

# Storage container names (allow override to keep code & infra in sync)
variable "uploads_container_name" {
  description = "Name of the uploads storage container"
  type        = string
  default     = "uploads"
}

variable "processed_container_name" {
  description = "Name of the processed videos storage container"
  type        = string
  default     = "processed-videos"
}

variable "queue_name" {
  description = "Name of the Service Bus queue for video processing jobs"
  type        = string
  default     = "video-processing-jobs"
}

# AI configuration
variable "speech_services_endpoint" {
  description = "Azure AI Speech Services endpoint"
  type        = string
  default     = ""
}

variable "speech_services_region" {
  description = "Azure AI Speech Services region"
  type        = string
  default     = ""
}

variable "openai_endpoint" {
  description = "Azure OpenAI endpoint"
  type        = string
  default     = ""
}

variable "openai_api_version" {
  description = "Azure OpenAI API version"
  type        = string
  default     = "2024-02-15-preview"
}

variable "openai_chat_deployment" {
  description = "Azure OpenAI chat deployment for cleanup"
  type        = string
  default     = ""
}

variable "openai_chat_model_name" {
  description = "Azure OpenAI chat model name for cleanup (e.g., 'gpt-4.1-nano')"
  type        = string
  default     = ""
}

variable "openai_chat_model_version" {
  description = "Azure OpenAI chat model version"
  type        = string
  default     = ""
}

variable "openai_api_key" {
  description = "Azure OpenAI API key"
  type        = string
  sensitive   = true
}

variable "speech_services_key" {
  description = "Azure AI Speech Services key"
  type        = string
  sensitive   = true
}

variable "translator_region" {
  description = "Azure Translator region"
  type        = string
  default     = ""
}

variable "translator_key" {
  description = "Azure Translator key"
  type        = string
  sensitive   = true
}

variable "streaming_min_replicas" {
  description = "Min replicas for streaming container app"
  type        = number
  default     = 0
}

variable "streaming_max_replicas" {
  description = "Max replicas for streaming container app"
  type        = number
  default     = 3
}

variable "streaming_image_name" {
  description = "Streaming container image name"
  type        = string
  default     = "audio-cleaner-streaming"
}

variable "streaming_image_tag" {
  description = "Streaming container image tag"
  type        = string
  default     = "latest"
}

variable "stream_session_signing_key" {
  description = "HMAC signing key for streaming session tokens"
  type        = string
  sensitive   = true
}
