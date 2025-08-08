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

variable "frontend_url" {
  description = "Frontend URL for Stripe redirects"
  type        = string
  default     = ""
}
