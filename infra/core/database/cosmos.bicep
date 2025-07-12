@description('Cosmos DB account name')
param cosmosAccountName string

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Tags to apply to all resources')
param tags object = {}

@description('The name for the database')
param databaseName string = 'audiocleaner'

resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: cosmosAccountName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    databaseAccountOfferType: 'Standard'
    enableAutomaticFailover: false
    enableMultipleWriteLocations: false
    enableFreeTier: false
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
    publicNetworkAccess: 'Enabled'
    networkAclBypass: 'AzureServices'
    disableKeyBasedMetadataWriteAccess: false
    enableAnalyticalStorage: false
    analyticalStorageConfiguration: {
      schemaType: 'WellDefined'
    }
    createMode: 'Default'
  }
}

resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosAccount
  name: databaseName
  properties: {
    resource: {
      id: databaseName
    }
    // No throughput options needed for serverless
  }
}

resource subscriptionsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDatabase
  name: 'subscriptions'
  properties: {
    resource: {
      id: 'subscriptions'
      partitionKey: {
        paths: ['/id']
        kind: 'Hash'
      }
      defaultTtl: -1
    }
    // No throughput options - inherits from database autoscale
  }
}

resource jobsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDatabase
  name: 'jobs'
  properties: {
    resource: {
      id: 'jobs'
      partitionKey: {
        paths: ['/id']
        kind: 'Hash'
      }
      defaultTtl: -1
    }
    // No throughput options - inherits from database autoscale
  }
}

resource rateLimitsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDatabase
  name: 'ratelimits'
  properties: {
    resource: {
      id: 'ratelimits'
      partitionKey: {
        paths: ['/clientId']
        kind: 'Hash'
      }
      defaultTtl: 3600 // 1 hour TTL for rate limit records
    }
  }
}

resource securityEventsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDatabase
  name: 'securityevents'
  properties: {
    resource: {
      id: 'securityevents'
      partitionKey: {
        paths: ['/clientId']
        kind: 'Hash'
      }
      defaultTtl: 86400 // 24 hours TTL for security events
    }
  }
}

output cosmosAccountName string = cosmosAccount.name
output cosmosAccountId string = cosmosAccount.id
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output cosmosDatabaseName string = databaseName
