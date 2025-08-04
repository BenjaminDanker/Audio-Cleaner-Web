// Shared pricing configuration based on actual Azure costs
// Fixed costs: ~$25/month (Cosmos DB free tier, Container Registry $5, Static Web App $9, Service Bus $10)
// Container Apps: Scale to zero when idle, ~$0.11/hour when processing
// Processing time: ~30 seconds per minute of audio = ~$0.001 compute cost per job
export const PRICING_CONFIG = {
  COST_PER_MB_CENTS: 0.05, // $0.0005 per MB = ~$0.50 per GB
  MINIMUM_COST_CENTS: 5, // $0.05 minimum (covers small files)
  CURRENCY: 'usd'
}

export const calculateJobCost = (fileSizeBytes) => {
  if (!fileSizeBytes || fileSizeBytes <= 0) {
    return 0
  }
  
  // Convert bytes to MB
  const fileSizeInMB = fileSizeBytes / (1024 * 1024)
  
  const cost = Math.max(PRICING_CONFIG.MINIMUM_COST_CENTS, fileSizeInMB * PRICING_CONFIG.COST_PER_MB_CENTS)
  
  // Round to 2 decimal places (in cents)
  return Math.round(cost)
}
