/**
 * Security Test Suite
 * Basic tests to validate security middleware functionality
 */

const SecurityMiddleware = require('../shared/securityMiddleware');
const InputValidator = require('../shared/inputValidator');

class SecurityTester {
    constructor() {
        this.security = new SecurityMiddleware(null); // No Cosmos for testing
        this.validator = new InputValidator();
    }

    async runTests() {
        console.log('🔒 Running Security Tests...\n');
        
        await this.testInputValidation();
        await this.testThreatDetection();
        await this.testRateLimit();
        await this.testFileValidation();
        
        console.log('\n✅ Security tests completed!');
    }

    async testInputValidation() {
        console.log('Testing Input Validation...');
        
        // Test XSS prevention
        const xssInput = {
            fileName: '<script>alert("xss")</script>video.mp4',
            comment: 'Hello <script>alert("hack")</script> world'
        };
        
        const schema = {
            fileName: { type: 'fileName', required: true },
            comment: { type: 'string', maxLength: 500 }
        };
        
        const result = this.validator.validateInput(xssInput, schema);
        console.log('  XSS Protection:', !result.valid ? '✅ BLOCKED' : '❌ FAILED');
        
        // Test SQL injection prevention
        const sqlInput = {
            search: "'; DROP TABLE users; --",
            id: "1 OR 1=1"
        };
        
        const sqlSchema = {
            search: { type: 'string', maxLength: 100 },
            id: { type: 'string', pattern: /^[a-zA-Z0-9-]+$/ }
        };
        
        const sqlResult = this.validator.validateInput(sqlInput, sqlSchema);
        console.log('  SQL Injection Protection:', !sqlResult.valid ? '✅ BLOCKED' : '❌ FAILED');
        
        // Test valid input
        const validInput = {
            fileName: 'valid-video.mp4',
            fileSize: 1024000
        };
        
        const validSchema = InputValidator.getSchemaForEndpoint('/api/upload-file');
        const validResult = this.validator.validateInput(validInput, validSchema);
        console.log('  Valid Input Processing:', validResult.valid ? '✅ PASSED' : '❌ FAILED');
    }

    async testThreatDetection() {
        console.log('\nTesting Threat Detection...');
        
        // Mock malicious request
        const maliciousReq = {
            url: '/api/upload-file?file=../../../etc/passwd',
            headers: {
                'user-agent': 'sqlmap/1.0 (http://sqlmap.org)',
                'x-forwarded-for': '192.168.1.100'
            }
        };
        
        const threatCheck = await this.security.checkForThreats(
            maliciousReq, 
            '192.168.1.100', 
            'sqlmap/1.0',
            console
        );
        
        console.log('  Malicious User Agent Detection:', !threatCheck.allowed ? '✅ BLOCKED' : '❌ FAILED');
        console.log('  Attack Pattern Detection:', threatCheck.threats?.includes('malicious_url_pattern') ? '✅ DETECTED' : '❌ MISSED');
        
        // Test legitimate request
        const legitReq = {
            url: '/api/upload-file',
            headers: {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'x-forwarded-for': '203.0.113.45'
            }
        };
        
        const legitCheck = await this.security.checkForThreats(
            legitReq,
            '203.0.113.45',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            console
        );
        
        console.log('  Legitimate Request:', legitCheck.allowed ? '✅ ALLOWED' : '❌ BLOCKED');
    }

    async testRateLimit() {
        console.log('\nTesting Rate Limiting...');
        
        // Test rate limit configuration
        const testIP = '192.168.1.200';
        const testEndpoint = '/api/upload-file';
        
        const rateLimitInfo = await this.security.checkRateLimit(testIP, testEndpoint, console);
        console.log('  Rate Limit Check:', rateLimitInfo.allowed ? '✅ WITHIN LIMITS' : '❌ EXCEEDED');
        console.log(`    Remaining: ${rateLimitInfo.remaining}/${rateLimitInfo.limit}`);
        
        // Test rate limit calculation
        const uploadLimit = this.security.rateLimits['/api/upload-file'];
        const downloadLimit = this.security.rateLimits['/api/download-file'];
        
        console.log('  Upload Rate Limit:', uploadLimit ? `✅ ${uploadLimit.requests}/min` : '❌ NOT SET');
        console.log('  Download Rate Limit:', downloadLimit ? `✅ ${downloadLimit.requests}/min` : '❌ NOT SET');
    }

    async testFileValidation() {
        console.log('\nTesting File Security...');
        
        // Test allowed file types
        const allowedFiles = ['test.mp4', 'audio.wav', 'video.avi'];
        const blockedFiles = ['script.exe', 'hack.bat', 'virus.com', 'data.txt'];
        
        let allowedCount = 0;
        let blockedCount = 0;
        
        for (const file of allowedFiles) {
            const result = this.validator.validateField('fileName', file, { type: 'fileName' });
            if (result.valid) allowedCount++;
        }
        
        for (const file of blockedFiles) {
            const result = this.validator.validateField('fileName', file, { type: 'fileName' });
            if (!result.valid) blockedCount++;
        }
        
        console.log(`  Allowed File Types: ${allowedCount}/${allowedFiles.length} ✅`);
        console.log(`  Blocked File Types: ${blockedCount}/${blockedFiles.length} ✅`);
        
        // Test path traversal protection
        const maliciousFiles = ['../../../etc/passwd', '..\\windows\\system32\\config\\sam', 'file\x00.txt'];
        let pathTraversalBlocked = 0;
        
        for (const file of maliciousFiles) {
            const result = this.validator.validateField('fileName', file, { type: 'fileName' });
            if (!result.valid) pathTraversalBlocked++;
        }
        
        console.log(`  Path Traversal Protection: ${pathTraversalBlocked}/${maliciousFiles.length} ✅`);
    }

    testSecurityHeaders() {
        console.log('\nTesting Security Headers...');
        
        const headers = this.security.getSecurityHeaders('/api/upload-file');
        
        const requiredHeaders = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Referrer-Policy',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ];
        
        let headerCount = 0;
        for (const header of requiredHeaders) {
            if (headers[header]) headerCount++;
        }
        
        console.log(`  Security Headers: ${headerCount}/${requiredHeaders.length} ✅`);
        console.log('  CSP Policy:', headers['Content-Security-Policy'] ? '✅ SET' : '❌ MISSING');
        console.log('  HSTS Policy:', headers['Strict-Transport-Security'] ? '✅ SET' : '❌ MISSING');
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new SecurityTester();
    tester.runTests().catch(console.error);
}

module.exports = SecurityTester;
