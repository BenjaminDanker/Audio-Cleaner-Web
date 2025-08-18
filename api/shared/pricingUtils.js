// Pricing utility for backend cost calculation
// Based on your original $0.50/GB pricing plus actual Azure service costs:
// 
// Pipeline:
// 1. Base processing: $0.50/GB (your original pricing)
// 2. Azure AI Speech Services (batch): $0.18/hour (vs $0.36/hour for OpenAI Whisper)
// 3. GPT-4.1-nano cleanup: ~$0.001/minute (token-based, minimal usage)
// 4. Azure Translator: $10/million characters (~2000 chars/minute = $0.02/minute per language)

const COST_PER_GB = 0.50; // $0.50 per GB as originally specified
const SPEECH_SERVICES_COST_PER_HOUR = 0.18; // Azure AI Speech Services batch transcription
const CLEANUP_COST_PER_MINUTE = 0.001; // GPT-4.1-nano for fixing transcription errors  
const TRANSLATOR_COST_PER_MINUTE_PER_LANG = 0.02; // Azure Translator: ~2000 chars/min @ $10/M chars
const MIN_CHARGE = 0.05; // $0.05 minimum (covers small files)

/**
 * Calculate the cost for processing based on file size and languages
 * For now estimates duration from file size, but should measure actual audio duration
 * @param {number} fileSizeInBytes - Size of the file in bytes
 * @param {Array} languagesRequested - Array of language codes requested
 * @returns {number} Cost in USD
 */
function calculateProcessingCost(fileSizeInBytes, languagesRequested = []) {
    if (!fileSizeInBytes || fileSizeInBytes <= 0) {
        return MIN_CHARGE;
    }
    
    // Base cost: $0.50 per GB
    const fileSizeInGB = fileSizeInBytes / (1024 * 1024 * 1024);
    let totalCost = fileSizeInGB * COST_PER_GB;
    
    // Estimate duration from file size (1MB ≈ 1 minute compressed audio)
    // TODO: Should measure actual audio duration instead of estimating
    const estimatedMinutes = Math.max(1, fileSizeInBytes / (1024 * 1024));
    const estimatedHours = estimatedMinutes / 60;
    
    // Add Azure service costs
    totalCost += estimatedHours * SPEECH_SERVICES_COST_PER_HOUR; // Azure AI Speech Services batch
    totalCost += estimatedMinutes * CLEANUP_COST_PER_MINUTE; // GPT-4.1-nano cleanup
    
    // Translation cost for additional languages (first language is transcribed, not translated)
    const additionalLanguages = Math.max(0, (languagesRequested?.length || 1) - 1);
    totalCost += additionalLanguages * TRANSLATOR_COST_PER_MINUTE_PER_LANG * estimatedMinutes;
    
    // Apply minimum charge
    totalCost = Math.max(MIN_CHARGE, totalCost);
    
    // Round to 2 decimal places
    return Math.round(totalCost * 100) / 100;
}

/**
 * Get breakdown of costs for transparency
 * @param {number} fileSizeInBytes - File size in bytes
 * @param {Array} languagesRequested - Array of language codes
 * @returns {Object} Cost breakdown
 */
function getCostBreakdown(fileSizeInBytes, languagesRequested = []) {
    const fileSizeInGB = fileSizeInBytes / (1024 * 1024 * 1024);
    const estimatedMinutes = Math.max(1, fileSizeInBytes / (1024 * 1024));
    const estimatedHours = estimatedMinutes / 60;
    const additionalLanguages = Math.max(0, (languagesRequested?.length || 1) - 1);
    
    return {
        baseCost: fileSizeInGB * COST_PER_GB,
        speechServicesCost: estimatedHours * SPEECH_SERVICES_COST_PER_HOUR,
        cleanupCost: estimatedMinutes * CLEANUP_COST_PER_MINUTE,
        translationCost: additionalLanguages * TRANSLATOR_COST_PER_MINUTE_PER_LANG * estimatedMinutes,
        totalCost: calculateProcessingCost(fileSizeInBytes, languagesRequested),
        estimatedMinutes: estimatedMinutes,
        additionalLanguages: additionalLanguages
    };
}

module.exports = {
    calculateProcessingCost,
    getCostBreakdown,
    COST_PER_GB,
    SPEECH_SERVICES_COST_PER_HOUR,
    TRANSLATOR_COST_PER_MINUTE_PER_LANG,
    MIN_CHARGE
};
