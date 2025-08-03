/**
 * Input Validation and Sanitization Helper
 * Provides comprehensive validation for all input types
 */

const validator = require('validator'); // Note: Add this to package.json

class InputValidator {
    constructor() {
        // Centralized limits - single source of truth
        this.maxSizes = {
            string: 10000,
            fileName: 255,
            email: 320,
            url: 2048,
            fileUpload: 5 * 1024 * 1024 * 1024 // 5GB - unified file upload limit
        };

        this.patterns = {
            fileName: /^[a-zA-Z0-9._-]+$/,
            jobId: /^job-\d+-[a-z0-9]+$/,
            guid: /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
            alphanumeric: /^[a-zA-Z0-9]+$/,
            safeString: /^[a-zA-Z0-9\s._-]+$/
        };

        this.allowedFileTypes = [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
            '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a'
        ];

        this.dangerousPatterns = [
            // Script injection
            /<script[^>]*>.*?<\/script>/gi,
            /javascript:/gi,
            /vbscript:/gi,
            /on\w+\s*=/gi,
            
            // SQL injection
            /(\b(union|select|insert|update|delete|drop|exec|execute|alter|create)\b.*\b(from|where|into|values|set|table)\b)/gi,
            /'.*'.*(\bor\b|\band\b).*'.*'/gi,
            /;.*drop\b/gi,
            
            // Command injection
            /(\||&|;|`|\$\(.*\))/g,
            /(nc|netcat|wget|curl|chmod|rm|mv|cp)\s/gi,
            
            // Path traversal
            /\.\.\/|\.\.\\|\.\.\//g,
            /\/etc\/passwd|\/etc\/shadow|\/proc\/|\/sys\//gi,
            
            // Template injection
            /\{\{.*\}\}|\$\{.*\}/g,
            /%\{.*\}|#\{.*\}/g
        ];
    }

    /**
     * Comprehensive input validation
     */
    validateInput(data, schema = {}) {
        const errors = [];
        const sanitized = {};

        if (!data || typeof data !== 'object') {
            return { valid: false, errors: ['Invalid input format'], data: null };
        }

        for (const [key, value] of Object.entries(data)) {
            const fieldSchema = schema[key] || { type: 'string', required: false };
            const validation = this.validateField(key, value, fieldSchema);
            
            if (!validation.valid) {
                errors.push(...validation.errors);
            } else {
                sanitized[key] = validation.sanitized;
            }
        }

        // Check for required fields
        for (const [key, fieldSchema] of Object.entries(schema)) {
            if (fieldSchema.required && !(key in data)) {
                errors.push(`Required field missing: ${key}`);
            }
        }

        return {
            valid: errors.length === 0,
            errors,
            data: errors.length === 0 ? sanitized : null
        };
    }

    /**
     * Validate individual field
     */
    validateField(fieldName, value, schema) {
        const errors = [];
        let sanitized = value;

        // Null/undefined checks
        if (value === null || value === undefined) {
            if (schema.required) {
                errors.push(`${fieldName} is required`);
            }
            return { valid: errors.length === 0, errors, sanitized: null };
        }

        // Type validation
        switch (schema.type) {
            case 'string':
                sanitized = this.validateString(fieldName, value, schema, errors);
                break;
            case 'number':
                sanitized = this.validateNumber(fieldName, value, schema, errors);
                break;
            case 'integer':
                sanitized = this.validateInteger(fieldName, value, schema, errors);
                break;
            case 'boolean':
                sanitized = this.validateBoolean(fieldName, value, schema, errors);
                break;
            case 'email':
                sanitized = this.validateEmail(fieldName, value, schema, errors);
                break;
            case 'url':
                sanitized = this.validateUrl(fieldName, value, schema, errors);
                break;
            case 'fileName':
                sanitized = this.validateFileName(fieldName, value, schema, errors);
                break;
            case 'guid':
                sanitized = this.validateGuid(fieldName, value, schema, errors);
                break;
            case 'jobId':
                sanitized = this.validateJobId(fieldName, value, schema, errors);
                break;
            default:
                sanitized = this.validateString(fieldName, value, schema, errors);
        }

        return { valid: errors.length === 0, errors, sanitized };
    }

    /**
     * String validation with sanitization
     */
    validateString(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        // Length checks
        const maxLength = schema.maxLength || this.maxSizes.string;
        const minLength = schema.minLength || 0;

        if (value.length > maxLength) {
            errors.push(`${fieldName} exceeds maximum length of ${maxLength}`);
            return value;
        }

        if (value.length < minLength) {
            errors.push(`${fieldName} is below minimum length of ${minLength}`);
            return value;
        }

        // Pattern validation
        if (schema.pattern && !schema.pattern.test(value)) {
            errors.push(`${fieldName} has invalid format`);
            return value;
        }

        // Dangerous pattern detection
        if (!schema.allowDangerous) {
            for (const pattern of this.dangerousPatterns) {
                if (pattern.test(value)) {
                    errors.push(`${fieldName} contains potentially dangerous content`);
                    return value;
                }
            }
        }

        // Sanitization
        let sanitized = value;
        
        if (schema.trim !== false) {
            sanitized = sanitized.trim();
        }

        if (schema.escape) {
            sanitized = this.escapeHtml(sanitized);
        }

        if (schema.normalize) {
            sanitized = sanitized.normalize('NFC');
        }

        return sanitized;
    }

    /**
     * Number validation
     */
    validateNumber(fieldName, value, schema, errors) {
        const num = typeof value === 'string' ? parseFloat(value) : value;
        
        if (typeof num !== 'number' || isNaN(num)) {
            errors.push(`${fieldName} must be a valid number`);
            return value;
        }

        if (schema.min !== undefined && num < schema.min) {
            errors.push(`${fieldName} must be at least ${schema.min}`);
            return value;
        }

        if (schema.max !== undefined && num > schema.max) {
            errors.push(`${fieldName} must be at most ${schema.max}`);
            return value;
        }

        return num;
    }

    /**
     * Integer validation
     */
    validateInteger(fieldName, value, schema, errors) {
        const num = typeof value === 'string' ? parseInt(value, 10) : value;
        
        if (!Number.isInteger(num)) {
            errors.push(`${fieldName} must be a valid integer`);
            return value;
        }

        if (schema.min !== undefined && num < schema.min) {
            errors.push(`${fieldName} must be at least ${schema.min}`);
            return value;
        }

        if (schema.max !== undefined && num > schema.max) {
            errors.push(`${fieldName} must be at most ${schema.max}`);
            return value;
        }

        return num;
    }

    /**
     * Boolean validation
     */
    validateBoolean(fieldName, value, schema, errors) {
        if (typeof value === 'boolean') {
            return value;
        }

        if (typeof value === 'string') {
            const lower = value.toLowerCase();
            if (lower === 'true' || lower === '1') return true;
            if (lower === 'false' || lower === '0') return false;
        }

        if (typeof value === 'number') {
            if (value === 1) return true;
            if (value === 0) return false;
        }

        errors.push(`${fieldName} must be a valid boolean`);
        return value;
    }

    /**
     * Email validation
     */
    validateEmail(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        if (value.length > this.maxSizes.email) {
            errors.push(`${fieldName} exceeds maximum email length`);
            return value;
        }

        // Use validator library for robust email validation
        if (typeof validator !== 'undefined' && !validator.isEmail(value)) {
            errors.push(`${fieldName} is not a valid email address`);
            return value;
        }

        // Basic regex fallback if validator not available
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (typeof validator === 'undefined' && !emailRegex.test(value)) {
            errors.push(`${fieldName} is not a valid email address`);
            return value;
        }

        return value.toLowerCase().trim();
    }

    /**
     * URL validation
     */
    validateUrl(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        if (value.length > this.maxSizes.url) {
            errors.push(`${fieldName} URL too long`);
            return value;
        }

        // Use validator library for robust URL validation
        if (typeof validator !== 'undefined' && !validator.isURL(value, { protocols: ['http', 'https'] })) {
            errors.push(`${fieldName} is not a valid URL`);
            return value;
        }

        // Basic validation fallback
        try {
            const url = new URL(value);
            if (!['http:', 'https:'].includes(url.protocol)) {
                errors.push(`${fieldName} must use HTTP or HTTPS protocol`);
                return value;
            }
        } catch {
            errors.push(`${fieldName} is not a valid URL`);
            return value;
        }

        return value;
    }

    /**
     * File name validation
     */
    validateFileName(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        if (value.length > this.maxSizes.fileName) {
            errors.push(`${fieldName} exceeds maximum filename length`);
            return value;
        }

        // Check for dangerous path traversal patterns
        if (value.includes('..') || value.includes('\\')) {
            errors.push(`${fieldName} contains invalid path characters`);
            return value;
        }

        // Check for null bytes and control characters
        if (/[\x00-\x1f\x7f-\x9f]/.test(value)) {
            errors.push(`${fieldName} contains invalid characters`);
            return value;
        }

        // Split by forward slash to validate each path segment individually
        const pathSegments = value.split('/');
        for (const segment of pathSegments) {
            if (segment === '' || segment === '.' || segment === '..') {
                errors.push(`${fieldName} contains invalid path segment`);
                return value;
            }
            
            // Each segment should only contain safe characters (no control chars, etc.)
            if (!/^[a-zA-Z0-9._-]+$/.test(segment)) {
                errors.push(`${fieldName} contains invalid characters in path segment: ${segment}`);
                return value;
            }
        }

        // Check file extension on the final segment (actual filename)
        const finalSegment = pathSegments[pathSegments.length - 1];
        const ext = finalSegment.toLowerCase().substring(finalSegment.lastIndexOf('.'));
        if (ext && !this.allowedFileTypes.includes(ext)) {
            errors.push(`${fieldName} has unsupported file type: ${ext}`);
            return value;
        }

        return value.trim();
    }

    /**
     * GUID validation
     */
    validateGuid(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        if (!this.patterns.guid.test(value)) {
            errors.push(`${fieldName} is not a valid GUID`);
            return value;
        }

        return value.toLowerCase();
    }

    /**
     * Job ID validation
     */
    validateJobId(fieldName, value, schema, errors) {
        if (typeof value !== 'string') {
            errors.push(`${fieldName} must be a string`);
            return value;
        }

        if (!this.patterns.jobId.test(value)) {
            errors.push(`${fieldName} is not a valid job ID format`);
            return value;
        }

        return value;
    }

    /**
     * HTML escaping
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }

    /**
     * Get file upload size limit
     */
    getFileUploadLimit() {
        return this.maxSizes.fileUpload;
    }

    /**
     * Create validation schema for common API endpoints
     */
    getSchemaForEndpoint(endpoint) {
        const schemas = {
            '/api/upload-file': {
                fileName: { type: 'fileName', required: true, maxLength: 255 },
                fileSize: { type: 'integer', min: 1, max: this.maxSizes.fileUpload } // Use centralized limit
            },
            '/api/enqueue-job': {
                fileName: { type: 'fileName', required: true },
                fileUrl: { type: 'url', required: true },
                processingType: { type: 'string', pattern: /^(denoise|enhance|normalize)$/, required: false },
                attenuationDb: { type: 'integer', min: 1, max: 100, required: false }
            },
            '/api/job-status': {
                jobId: { type: 'jobId', required: true }
            },
            '/api/download-file': {
                filename: { type: 'fileName', required: true }
            }
        };

        return schemas[endpoint] || {};
    }

    /**
     * Quick sanitize for simple cases
     */
    static sanitizeString(input, maxLength = 1000) {
        if (typeof input !== 'string') return '';
        
        return input
            .trim()
            .substring(0, maxLength)
            .replace(/[<>]/g, '') // Remove potential HTML
            .normalize('NFC');
    }
}

module.exports = InputValidator;
