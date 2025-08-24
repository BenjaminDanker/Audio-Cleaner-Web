// Shared pricing configuration based on your original $0.50/GB plus Azure service costs
// 
// Pipeline:
// 1. Base processing: $0.50/GB (your original pricing)
// 2. Azure AI Speech Services (batch): $0.18/hour (vs $0.36/hour for OpenAI Whisper)
// 3. GPT-4.1-nano cleanup: ~$0.001/minute (token-based, minimal usage)
// 4. Azure Translator: $10/million characters (~2000 chars/minute = $0.02/minute per language)

export const PRICING_CONFIG = {
  COST_PER_GB: 0.50, // $0.50 per GB as originally specified
  SPEECH_SERVICES_COST_PER_HOUR: 0.18, // Azure AI Speech Services batch transcription
  CLEANUP_COST_PER_MINUTE: 0.001, // GPT-4.1-nano for fixing transcription errors
  TRANSLATOR_COST_PER_MINUTE_PER_LANG: 0.02, // Azure Translator: ~2000 chars/min @ $10/M chars
  MIN_CHARGE: 0.05, // $0.05 minimum charge
  CURRENCY: 'usd'
}

/**
 * Calculate job cost based on file size and languages
 * @param {number} fileSizeBytes - File size in bytes
 * @param {Array} selectedLanguages - Array of selected language codes
 * @returns {number} Cost in cents
 */
export const calculateJobCost = (fileSizeBytes, selectedLanguages = []) => {
  if (!fileSizeBytes || fileSizeBytes <= 0) {
    return Math.round(PRICING_CONFIG.MIN_CHARGE * 100) // Return in cents
  }
  
  // Base cost: $0.50 per GB
  const fileSizeInGB = fileSizeBytes / (1024 * 1024 * 1024)
  let totalCost = fileSizeInGB * PRICING_CONFIG.COST_PER_GB
  
  // Estimate duration from file size (1MB ≈ 1 minute compressed audio)
  // TODO: Should measure actual audio duration instead of estimating
  const estimatedMinutes = Math.max(1, fileSizeBytes / (1024 * 1024))
  const estimatedHours = estimatedMinutes / 60
  
  // If no languages selected, skip transcription/cleanup/translation costs
  const hasSubtitles = Array.isArray(selectedLanguages) && selectedLanguages.length > 0
  if (hasSubtitles) {
    // Add Azure service costs for subtitles
    totalCost += estimatedHours * PRICING_CONFIG.SPEECH_SERVICES_COST_PER_HOUR // Azure AI Speech Services batch
    totalCost += estimatedMinutes * PRICING_CONFIG.CLEANUP_COST_PER_MINUTE // GPT-4.1-nano cleanup
    
    // Translation cost applies per selected language (including the first)
    const translationLanguages = selectedLanguages?.length || 0
    totalCost += translationLanguages * PRICING_CONFIG.TRANSLATOR_COST_PER_MINUTE_PER_LANG * estimatedMinutes
  }
  
  // Apply minimum charge
  totalCost = Math.max(PRICING_CONFIG.MIN_CHARGE, totalCost)
  
  // Return cost in cents
  return Math.round(totalCost * 100)
}

/**
 * Format cost for display
 * @param {number} costInCents - Cost in cents
 * @returns {string} Formatted cost string
 */
export const formatCost = (costInCents) => {
  const dollars = costInCents / 100
  return `$${dollars.toFixed(2)}`
}
