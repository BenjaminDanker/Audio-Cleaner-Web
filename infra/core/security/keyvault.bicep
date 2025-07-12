@description('Name for the Key Vault')
param name string

@description('Location for the Key Vault')
param location string = resourceGroup().location

@description('Tags to apply to the Key Vault')
param tags object = {}

@description('Tenant ID for the Key Vault')
param tenantId string = tenant().tenantId

@description('Object IDs of users/groups that should have access to Key Vault')
param keyVaultAdminObjectIds array = []

@description('Enable network restrictions')
param enableNetworkRestrictions bool = true

@description('Allowed IP ranges for Key Vault access')
param allowedIpRanges array = []

@description('Log Analytics workspace ID for diagnostic settings')
param logAnalyticsWorkspaceId string

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: tenantId
    sku: {
      family: 'A'
      name: 'premium' // Enhanced to premium for HSM support
    }
    accessPolicies: []
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enableRbacAuthorization: true
    enablePurgeProtection: true
    createMode: 'default'
    networkAcls: enableNetworkRestrictions ? {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      ipRules: map(allowedIpRanges, ipRange => {
        value: ipRange
      })
      virtualNetworkRules: []
    } : {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
  }
}

// Create diagnostic settings for security monitoring
resource keyVaultDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: '${name}-diagnostics'
  scope: keyVault
  properties: {
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
    workspaceId: logAnalyticsWorkspaceId
  }
}

// Role assignments for Key Vault administrators
resource keyVaultAdminRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for objectId in keyVaultAdminObjectIds: {
  name: guid(keyVault.id, objectId, 'Key Vault Administrator')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483') // Key Vault Administrator
    principalId: objectId
    principalType: 'User'
  }
}]

output name string = keyVault.name
output id string = keyVault.id
output uri string = keyVault.properties.vaultUri
