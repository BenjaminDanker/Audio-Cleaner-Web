@description('Container App name')
param name string

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Tags to apply to all resources')
param tags object = {}

@description('Container Apps environment name')
param containerAppsEnvironmentName string

@description('Container registry name')
param containerRegistryName string

@description('Application Insights name')
param applicationInsightsName string

@description('Storage account name')
param storageAccountName string

@description('Service Bus namespace name')
param serviceBusNamespaceName string

@description('Cosmos DB account name')
param cosmosAccountName string

// Get existing resources
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: containerAppsEnvironmentName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: containerRegistryName
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: applicationInsightsName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' existing = {
  name: storageAccountName
}

resource serviceBusNamespace 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' existing = {
  name: serviceBusNamespaceName
}

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' existing = {
  name: cosmosAccountName
}

// Create managed identity for Container App
resource containerAppManagedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${name}-identity'
  location: location
  tags: tags
}

// Create Container App for audio processing
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: union(tags, {
    'azd-service-name': 'processor'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${containerAppManagedIdentity.id}': {}
    }
  }
  properties: {
    environmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8080
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
          allowCredentials: true
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: containerAppManagedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'storage-connection-string'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'servicebus-connection-string'
          value: listKeys('${serviceBusNamespace.id}/authorizationRules/RootManageSharedAccessKey', serviceBusNamespace.apiVersion).primaryConnectionString
        }
        {
          name: 'cosmos-connection-string'
          value: 'AccountEndpoint=https://${cosmosAccount.name}.documents.azure.com:443/;AccountKey=${cosmosAccount.listKeys().primaryMasterKey}'
        }
      ]
    }
    template: {
      scale: {
        minReplicas: 0
        maxReplicas: 5
        rules: [
          {
            name: 'servicebus-queue-length'
            custom: {
              type: 'azure-servicebus'
              metadata: {
                queueName: 'audio-processing-queue'
                messageCount: '5'
              }
              auth: [
                {
                  secretRef: 'servicebus-connection-string'
                  triggerParameter: 'connection'
                }
              ]
            }
          }
        ]
      }
      containers: [
        {
          name: 'audio-processor'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          env: [
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'AZURE_SERVICE_BUS_CONNECTION_STRING'
              secretRef: 'servicebus-connection-string'
            }
            {
              name: 'COSMOS_CONNECTION_STRING'
              secretRef: 'cosmos-connection-string'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: applicationInsights.properties.ConnectionString
            }
          ]
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
    }
  }
}

// Grant Container App access to ACR
resource acrPullRole 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull role
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerAppManagedIdentity.id, containerRegistry.id, acrPullRole.id)
  properties: {
    roleDefinitionId: acrPullRole.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Storage Blob Data Owner
resource storageBlobDataOwner 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b' // Storage Blob Data Owner
}

resource storageBlobOwnerRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(containerAppManagedIdentity.id, storageAccount.id, storageBlobDataOwner.id)
  properties: {
    roleDefinitionId: storageBlobDataOwner.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Storage Blob Data Contributor
resource storageBlobDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
}

resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(containerAppManagedIdentity.id, storageAccount.id, storageBlobDataContributor.id)
  properties: {
    roleDefinitionId: storageBlobDataContributor.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Service Bus
resource serviceBusDataReceiver 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '4f6d3b9b-027b-4f4c-9142-0e5a2a2247e0' // Azure Service Bus Data Receiver
}

resource serviceBusReceiverRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: serviceBusNamespace
  name: guid(containerAppManagedIdentity.id, serviceBusNamespace.id, serviceBusDataReceiver.id)
  properties: {
    roleDefinitionId: serviceBusDataReceiver.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Storage Queue Data
resource storageQueueDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '974c5e8b-45b9-4653-ba55-5f855dd0fb88' // Storage Queue Data Contributor
}

resource storageQueueRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(containerAppManagedIdentity.id, storageAccount.id, storageQueueDataContributor.id)
  properties: {
    roleDefinitionId: storageQueueDataContributor.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Storage Table Data
resource storageTableDataContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3' // Storage Table Data Contributor
}

resource storageTableRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(containerAppManagedIdentity.id, storageAccount.id, storageTableDataContributor.id)
  properties: {
    roleDefinitionId: storageTableDataContributor.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Container App access to Monitoring Metrics Publisher
resource monitoringMetricsPublisher 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '3913510d-42f4-4e42-8a64-420c390055eb' // Monitoring Metrics Publisher
}

resource monitoringMetricsRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: resourceGroup()
  name: guid(containerAppManagedIdentity.id, resourceGroup().id, monitoringMetricsPublisher.id)
  properties: {
    roleDefinitionId: monitoringMetricsPublisher.id
    principalId: containerAppManagedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

output containerAppName string = containerApp.name
output containerAppId string = containerApp.id
output containerAppManagedIdentityId string = containerAppManagedIdentity.id
