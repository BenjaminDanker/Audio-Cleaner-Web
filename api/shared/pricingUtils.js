// Pricing utility for backend cost calculation
// Based on actual Azure infrastructure costs:
// - Fixed: ~$25/month (Cosmos DB free, Container Registry $5, SWA $9, Service Bus $10)
// - Container Apps: Scale to zero, ~$0.11/hour when active
// - Processing: ~30 seconds per minute of audio = ~$0.001 per job
const COST_PER_MB = 0.0005; // $0.0005 per MB = ~$0.50 per GB

/**
 * Calculate the cost for processing based on file size
 * @param {number} fileSizeInBytes - Size of the file in bytes
 * @returns {number} Cost in USD
 */
function calculateProcessingCost(fileSizeInBytes) {
    if (!fileSizeInBytes || fileSizeInBytes <= 0) {
        return 0;
    }
    
    // Convert bytes to MB
    const fileSizeInMB = fileSizeInBytes / (1024 * 1024);
    
    // Calculate cost (minimum charge of $0.05 for any file)
    const cost = Math.max(0.05, fileSizeInMB * COST_PER_MB);
    
    // Round to 2 decimal places
    return Math.round(cost * 100) / 100;
}

/**
 * Get the current pricing rate
 * @returns {number} Cost per MB in USD
 */
function getCostPerMB() {
    return COST_PER_MB;
}

module.exports = {
    calculateProcessingCost,
    getCostPerMB,
    COST_PER_MB
};
