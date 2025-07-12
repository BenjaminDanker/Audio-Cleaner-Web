@description('Name for the storage account')
param name string

@description('Location for the storage account')
param location string = resourceGroup().location

@description('Tags to apply to the storage account')
param tags object = {}

@description('Storage account type')
@allowed([
  'Standard_LRS'
  'Standard_GRS'
  'Standard_RAGRS'
  'Standard_ZRS'
  'Premium_LRS'
  'Premium_ZRS'
  'Standard_GZRS'
  'Standard_RAGZRS'
])
param kind string = 'Standard_LRS'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: name
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: kind
  }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: true
    supportsHttpsTrafficOnly: true
    networkAcls: {
      bypass: 'AzureServices'
      virtualNetworkRules: []
      ipRules: []
      defaultAction: 'Allow'
    }
    accessTier: 'Hot'
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2022-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    cors: {
      corsRules: [
        {
          allowedOrigins: [
            'https://zealous-river-020499810.2.azurestaticapps.net'
            'http://localhost:3000'
            'http://localhost:5173'
            'http://127.0.0.1:3000'
            'http://127.0.0.1:5173'
          ]
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']
          allowedHeaders: ['*']
          exposedHeaders: [
            'Content-Range'
            'Content-Length'
            'Content-Type'
            'Accept-Ranges'
            'ETag'
            'Last-Modified'
            'x-ms-*'
          ]
          maxAgeInSeconds: 3600
        }
      ]
    }
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource uploadsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  parent: blobServices
  name: 'uploads'
  properties: {
    publicAccess: 'None'
  }
}

resource processedContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-05-01' = {
  parent: blobServices
  name: 'processed'
  properties: {
    publicAccess: 'None'
  }
}

// Lifecycle management policy to automatically clean up old blobs
resource lifecyclePolicy 'Microsoft.Storage/storageAccounts/managementPolicies@2022-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'cleanupUploads'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['uploads/']
            }
            actions: {
              baseBlob: {
                delete: {
                  daysAfterModificationGreaterThan: 7 // Delete uploads after 7 days
                }
              }
            }
          }
        }
        {
          name: 'cleanupProcessed'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['processed/']
            }
            actions: {
              baseBlob: {
                tierToCool: {
                  daysAfterModificationGreaterThan: 30 // Move to cool storage after 30 days
                }
                tierToArchive: {
                  daysAfterModificationGreaterThan: 90 // Move to archive after 90 days
                }
                delete: {
                  daysAfterModificationGreaterThan: 365 // Delete after 1 year
                }
              }
            }
          }
        }
        {
          name: 'cleanupOrphanedBlobs'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: ['blockBlob']
            }
            actions: {
              baseBlob: {
                delete: {
                  daysAfterModificationGreaterThan: 30 // Delete any blob older than 30 days
                }
              }
            }
          }
        }
      ]
    }
  }
}

output name string = storageAccount.name
output id string = storageAccount.id
