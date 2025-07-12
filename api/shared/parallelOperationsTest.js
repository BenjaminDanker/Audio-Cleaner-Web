/**
 * Enhanced Rate Limiting Test for Parallel Operations
 */

const SecurityMiddleware = require('./securityMiddleware');
const ParallelOperationsConfig = require('./parallelOperationsConfig');

class ParallelOperationsTest {
    constructor() {
        this.security = new SecurityMiddleware(null); // No Cosmos for testing
    }

    async runTests() {
        console.log('🔄 Testing Parallel Operations Rate Limiting...\n');
        
        await this.testParallelUploadDetection();
        await this.testDynamicRateLimiting();
        await this.testOptimalConcurrency();
        await this.testRateLimitRecovery();
        
        console.log('\n✅ Parallel operations tests completed!');
    }

    async testParallelUploadDetection() {
        console.log('Testing Parallel Operation Detection...');
        
        // Mock chunk upload request
        const chunkRequest = {
            headers: {
                'x-chunk-upload': 'true',
                'x-expected-chunks': '25',
                'x-upload-strategy': 'enhanced',
                'content-length': '4194304' // 4MB
            },
            query: { blockid: 'chunk_001' }
        };
        
        const detection = ParallelOperationsConfig.detectParallelOperation(chunkRequest);
        console.log('  Chunk Upload Detection:', detection.isChunkUpload ? '✅ DETECTED' : '❌ MISSED');
        console.log('  Expected Chunks:', detection.expectedChunks === 25 ? '✅ CORRECT' : '❌ INCORRECT');
        console.log('  Upload Strategy:', detection.uploadStrategy === 'enhanced' ? '✅ CORRECT' : '❌ INCORRECT');
        
        // Mock regular upload request
        const regularRequest = {
            headers: { 'content-length': '1048576' },
            query: {}
        };
        
        const regularDetection = ParallelOperationsConfig.detectParallelOperation(regularRequest);
        console.log('  Regular Upload Detection:', !regularDetection.isChunkUpload ? '✅ CORRECT' : '❌ INCORRECT');
    }

    async testDynamicRateLimiting() {
        console.log('\nTesting Dynamic Rate Limiting...');
        
        // Test small file (standard limits)
        const smallFileConfig = ParallelOperationsConfig.getUploadConfig(10 * 1024 * 1024); // 10MB
        const smallFileRateLimit = ParallelOperationsConfig.getRateLimitConfig(smallFileConfig.rateLimitStrategy);
        
        console.log('  Small File Strategy:', smallFileConfig.rateLimitStrategy === 'standard' ? '✅ CORRECT' : '❌ INCORRECT');
        console.log('  Small File Parallel:', !smallFileConfig.useParallel ? '✅ CORRECT' : '❌ INCORRECT');
        
        // Test large file (enhanced limits)  
        const largeFileConfig = ParallelOperationsConfig.getUploadConfig(200 * 1024 * 1024); // 200MB
        const largeFileRateLimit = ParallelOperationsConfig.getRateLimitConfig(largeFileConfig.rateLimitStrategy, true);
        
        console.log('  Large File Strategy:', largeFileConfig.rateLimitStrategy === 'enhanced' ? '✅ CORRECT' : '❌ INCORRECT');
        console.log('  Large File Parallel:', largeFileConfig.useParallel ? '✅ CORRECT' : '❌ INCORRECT');
        console.log('  Enhanced Rate Limit:', largeFileRateLimit.requests > smallFileRateLimit.requests ? '✅ HIGHER' : '❌ NOT HIGHER');
        
        // Test very large file (bulk limits)
        const xlFileConfig = ParallelOperationsConfig.getUploadConfig(1024 * 1024 * 1024); // 1GB
        const xlFileRateLimit = ParallelOperationsConfig.getRateLimitConfig(xlFileConfig.rateLimitStrategy, true);
        
        console.log('  XL File Strategy:', xlFileConfig.rateLimitStrategy === 'bulk' ? '✅ CORRECT' : '❌ INCORRECT');
        console.log('  XL File Burst Limit:', xlFileRateLimit.burstLimit > largeFileRateLimit.burstLimit ? '✅ HIGHER' : '❌ NOT HIGHER');
    }

    async testOptimalConcurrency() {
        console.log('\nTesting Optimal Concurrency Calculation...');
        
        // Test different file sizes
        const testCases = [
            { size: 50 * 1024 * 1024, expectedMax: 3 }, // 50MB
            { size: 200 * 1024 * 1024, expectedMax: 6 }, // 200MB  
            { size: 1024 * 1024 * 1024, expectedMax: 8 } // 1GB
        ];
        
        for (const testCase of testCases) {
            const config = ParallelOperationsConfig.getUploadConfig(testCase.size);
            const rateLimitConfig = ParallelOperationsConfig.getRateLimitConfig(config.rateLimitStrategy, true);
            const optimalConcurrency = ParallelOperationsConfig.calculateOptimalConcurrency(testCase.size, rateLimitConfig);
            
            const sizeLabel = testCase.size / (1024 * 1024) >= 1024 ? 
                `${testCase.size / (1024 * 1024 * 1024)}GB` : 
                `${testCase.size / (1024 * 1024)}MB`;
            
            console.log(`  ${sizeLabel} File - Optimal Concurrency: ${optimalConcurrency}/${config.maxConcurrency} ${optimalConcurrency <= config.maxConcurrency ? '✅' : '❌'}`);
        }
    }

    async testRateLimitRecovery() {
        console.log('\nTesting Rate Limit Recovery...');
        
        // Simulate rate limit scenario
        const testIP = '192.168.1.100';
        const testEndpoint = '/api/upload-file';
        const fileSize = 100 * 1024 * 1024; // 100MB
        
        try {
            // Test enhanced rate limiting for file operations
            const result1 = await this.security.checkFileOperationRateLimit(testIP, testEndpoint, {
                fileSize,
                isChunkUpload: true,
                userId: 'test-user-123'
            }, console);
            
            console.log('  First Check:', result1.allowed ? '✅ ALLOWED' : '❌ BLOCKED');
            console.log(`    Limit: ${result1.limit}, Remaining: ${result1.remaining}`);
            
            // Test with different rate limit type
            const result2 = await this.security.checkFileOperationRateLimit(testIP, testEndpoint, {
                fileSize: 10 * 1024 * 1024, // Small file
                isChunkUpload: false,
                userId: 'test-user-123'
            }, console);
            
            console.log('  Small File Check:', result2.allowed ? '✅ ALLOWED' : '❌ BLOCKED');
            console.log(`    Different Limits Applied: ${result2.limit !== result1.limit ? '✅ YES' : '❌ NO'}`);
            
        } catch (error) {
            console.log('  Rate Limit Test: ❌ ERROR -', error.message);
        }
    }

    testFrontendConfig() {
        console.log('\nTesting Frontend Configuration...');
        
        const testSizes = [
            32 * 1024 * 1024,   // 32MB
            128 * 1024 * 1024,  // 128MB
            512 * 1024 * 1024,  // 512MB
            2048 * 1024 * 1024  // 2GB
        ];
        
        for (const size of testSizes) {
            const config = ParallelOperationsConfig.getFrontendConfig(size);
            const sizeLabel = size >= 1024 * 1024 * 1024 ? 
                `${(size / (1024 * 1024 * 1024)).toFixed(1)}GB` : 
                `${Math.round(size / (1024 * 1024))}MB`;
            
            console.log(`  ${sizeLabel}:`);
            console.log(`    Parallel: ${config.useParallelUpload ? 'Yes' : 'No'}`);
            console.log(`    Concurrency: ${config.maxConcurrency}`);
            console.log(`    Chunk Size: ${config.chunkSize / (1024 * 1024)}MB`);
            console.log(`    Headers: ${Object.keys(config.headers).length} headers ✅`);
        }
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new ParallelOperationsTest();
    tester.runTests()
        .then(() => tester.testFrontendConfig())
        .catch(console.error);
}

module.exports = ParallelOperationsTest;
