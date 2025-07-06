targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Name of the resource group. If empty, a name will be generated.')
param resourceGroupName string = ''

// Optional parameters for customization
@description('Tags to apply to all resources')
param tags object = {}

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName))

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: union(tags, {
    'azd-env-name': environmentName
  })
}

module monitoring './core/monitor/monitoring.bicep' = {
  name: 'monitoring'
  scope: rg
  params: {
    location: location
    tags: tags
    logAnalyticsName: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: '${abbrs.insightsComponents}${resourceToken}'
  }
}

module storage './core/storage/storage-account.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    location: location
    tags: tags
    name: '${abbrs.storageStorageAccounts}${resourceToken}'
  }
}

module keyVault './core/security/keyvault.bicep' = {
  name: 'keyvault'
  scope: rg
  params: {
    location: location
    tags: tags
    name: '${abbrs.keyVaultVaults}${resourceToken}'
  }
}

module cosmos './core/database/cosmos.bicep' = {
  name: 'cosmos'
  scope: rg
  params: {
    location: location
    tags: tags
    cosmosAccountName: '${abbrs.documentDBDatabaseAccounts}${resourceToken}'
  }
}

module serviceBus './core/messaging/servicebus.bicep' = {
  name: 'servicebus'
  scope: rg
  params: {
    location: location
    tags: tags
    serviceBusNamespaceName: '${abbrs.serviceBusNamespaces}${resourceToken}'
  }
}

module functionApp './core/host/function-app.bicep' = {
  name: 'function-app'
  scope: rg
  params: {
    location: location
    tags: tags
    functionAppName: '${abbrs.webSitesFunctions}${resourceToken}'
    storageAccountName: storage.outputs.name
    applicationInsightsName: monitoring.outputs.applicationInsightsName
    keyVaultName: keyVault.outputs.name
    cosmosAccountName: cosmos.outputs.cosmosAccountName
    serviceBusNamespaceName: serviceBus.outputs.serviceBusNamespaceName
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
  }
}

module staticWebApp './core/host/static-web-app.bicep' = {
  name: 'static-web-app'
  scope: rg
  params: {
    location: location
    tags: tags
    staticWebAppName: '${abbrs.webStaticSites}${resourceToken}'
    functionAppUrl: functionApp.outputs.functionAppUrl
  }
}

module containerRegistry './core/host/container-registry.bicep' = {
  name: 'container-registry'
  scope: rg
  params: {
    name: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
  }
}

module containerAppsEnvironment './core/host/container-apps-environment.bicep' = {
  name: 'container-apps-environment'
  scope: rg
  params: {
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
  }
}

// Container App for the AI processing service
module processorContainerApp './app/processor.bicep' = {
  name: 'processor'
  scope: rg
  params: {
    name: '${abbrs.appContainerApps}processor-${resourceToken}'
    location: location
    tags: tags
    containerAppsEnvironmentName: containerAppsEnvironment.outputs.name
    containerRegistryName: containerRegistry.outputs.name
    applicationInsightsName: monitoring.outputs.applicationInsightsName
    storageAccountName: storage.outputs.name
    serviceBusNamespaceName: serviceBus.outputs.serviceBusNamespaceName
    cosmosAccountName: cosmos.outputs.cosmosAccountName
  }
}

// App outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = rg.name
output RESOURCE_GROUP_ID string = rg.id

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerRegistry.outputs.name

output AZURE_CONTAINER_APPS_ENVIRONMENT_NAME string = containerAppsEnvironment.outputs.name

output AZURE_FUNCTION_APP_NAME string = functionApp.outputs.functionAppName
output AZURE_FUNCTION_APP_URL string = functionApp.outputs.functionAppUrl

output AZURE_STATIC_WEB_APP_NAME string = staticWebApp.outputs.staticWebAppName
output AZURE_STATIC_WEB_APP_URL string = staticWebApp.outputs.staticWebAppUrl

output AZURE_STORAGE_ACCOUNT_NAME string = storage.outputs.name
output AZURE_KEY_VAULT_NAME string = keyVault.outputs.name
output AZURE_APPLICATION_INSIGHTS_NAME string = monitoring.outputs.applicationInsightsName
output AZURE_LOG_ANALYTICS_WORKSPACE_NAME string = monitoring.outputs.logAnalyticsWorkspaceName

output AZURE_COSMOS_ACCOUNT_NAME string = cosmos.outputs.cosmosAccountName
output AZURE_COSMOS_ENDPOINT string = cosmos.outputs.cosmosEndpoint

output AZURE_SERVICE_BUS_NAMESPACE_NAME string = serviceBus.outputs.serviceBusNamespaceName
output AZURE_SERVICE_BUS_ENDPOINT string = serviceBus.outputs.serviceBusEndpoint

// Add processor app name for your existing AZURE_PROCESSOR_APP_NAME variable
output AZURE_PROCESSOR_APP_NAME string = processorContainerApp.outputs.containerAppName
