// Create Service Bus namespace and queue
@description('Service Bus namespace name')
param serviceBusNamespaceName string

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Tags to apply to all resources')
param tags object = {}

resource serviceBusNamespace 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: serviceBusNamespaceName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  properties: {
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

resource serviceBusQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBusNamespace
  name: 'audio-processing-queue'
  properties: {
    lockDuration: 'PT2M'  // Reduced from 5M - audio processing should be faster or fail quickly
    maxSizeInMegabytes: 1024
    requiresDuplicateDetection: false
    requiresSession: false
    defaultMessageTimeToLive: 'P7D'  // Reduced from 14D - audio jobs shouldn't sit that long
    deadLetteringOnMessageExpiration: true  // Enable dead lettering for failed jobs
    maxDeliveryCount: 3  // Reduced for faster failure detection and lower costs
    // autoDeleteOnIdle: 'P30D'  // Removed - not supported in Basic tier
    enablePartitioning: false  // Not available in Basic tier
    enableExpress: false  // Not available in Basic tier
  }
}

// Output the connection string for use by Azure Functions
output serviceBusNamespaceName string = serviceBusNamespace.name
output serviceBusNamespaceId string = serviceBusNamespace.id
output serviceBusEndpoint string = serviceBusNamespace.properties.serviceBusEndpoint
