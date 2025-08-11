import React, { useState, useRef, useEffect } from 'react'
import { Upload, FileVideo, X, DollarSign, Clock } from 'lucide-react'
import axios from 'axios'
import { useAccount } from './AccountContext'
import { calculateJobCost } from '../utils/pricing'
import './VideoUpload.css'

// Configuration function for optimal upload settings
const getUploadConfig = (fileSize) => {
  if (fileSize < 64 * 1024 * 1024) { // < 64MB
    return {
      useParallel: false,
      maxConcurrency: 1,
      chunkSize: 4 * 1024 * 1024,
      retryDelay: 1000,
      rateLimitStrategy: 'standard'
    }
  } else if (fileSize < 500 * 1024 * 1024) { // 64MB - 500MB
    return {
      useParallel: true,
      maxConcurrency: 3, // Reduced to respect rate limits
      chunkSize: 4 * 1024 * 1024,
      retryDelay: 2000,
      rateLimitStrategy: 'enhanced'
    }
  } else if (fileSize < 2 * 1024 * 1024 * 1024) { // 500MB - 2GB
    return {
      useParallel: true,
      maxConcurrency: 4,
      chunkSize: 8 * 1024 * 1024, // Larger chunks for big files
      retryDelay: 3000,
      rateLimitStrategy: 'bulk'
    }
  } else { // > 2GB
    return {
      useParallel: true,
      maxConcurrency: 5,
      chunkSize: 16 * 1024 * 1024, // Even larger chunks
      retryDelay: 5000,
      rateLimitStrategy: 'enterprise'
    }
  }
}

// Utility function for parallel block uploads with Azure best practices
// Enhanced parallel upload with rate limiting awareness
const uploadFileInParallel = async (file, uploadUrl, onProgress, uploadId, activeUploadIdRef) => {
  const fileSize = file.size
  
  // Dynamic configuration based on file size
  const config = getUploadConfig(fileSize)
  
  // For files smaller than 64MB, use simple upload
  if (!config.useParallel) {
    return uploadFileSimple(file, uploadUrl, onProgress, uploadId, activeUploadIdRef)
  }
  
  const blockCount = Math.ceil(fileSize / config.chunkSize)
  const blockIds = []
  
  // Generate block IDs with proper encoding
  for (let i = 0; i < blockCount; i++) {
    // Use proper base64 encoding for Azure - must be exactly 64 bytes when base64 decoded
    const blockId = btoa(String.fromCharCode(...new Array(64).fill(0).map((_, idx) => 
      idx < 8 ? (i >>> ((7 - idx) * 8)) & 0xFF : 0
    )))
    blockIds.push(blockId)
  }
  
  let uploadedBytes = 0
  const totalBytes = fileSize
  const failedBlocks = new Set()
  const blockProgressMap = new Map() // Track individual block progress
  
  // Function to calculate and report overall progress
  const updateOverallProgress = () => {
    let totalProgress = 0
    for (const [blockIndex, progress] of blockProgressMap) {
      const blockSize = Math.min(config.chunkSize, totalBytes - (blockIndex * config.chunkSize))
      totalProgress += (progress * blockSize)
    }
    const overallProgress = Math.min(totalProgress / totalBytes, 1.0)
    
    // Check if upload is still active before calling progress callback
    if (activeUploadIdRef.current === uploadId) {
      onProgress(overallProgress)
    }
  }
  
  // Enhanced block upload with retry logic and smooth progress
  const uploadBlockWithRetry = async (blockIndex, maxRetries = 3) => {
    const blockId = blockIds[blockIndex]
    const start = blockIndex * config.chunkSize  // Use config.chunkSize instead of blockSize
    const end = Math.min(start + config.chunkSize, fileSize)
    const chunk = file.slice(start, end)
    
    // Initialize progress for this block
    blockProgressMap.set(blockIndex, 0)
    
    // Parse the upload URL to get base URL and SAS token
    const urlParts = uploadUrl.split('?')
    const baseUrl = urlParts[0]
    const sasToken = urlParts[1]
    
    const blockUrl = `${baseUrl}?comp=block&blockid=${encodeURIComponent(blockId)}&${sasToken}`
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          
          // Set longer timeout for larger files
          xhr.timeout = Math.min(240000, 60000 + (chunk.size / 1024 / 1024) * 5000) // 1-4 minutes based on chunk size
          
          // Track upload progress for this specific block
          xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
              const blockProgress = e.loaded / e.total
              blockProgressMap.set(blockIndex, blockProgress)
              updateOverallProgress()
            }
          })
          
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              // Ensure this block is marked as 100% complete
              blockProgressMap.set(blockIndex, 1.0)
              updateOverallProgress()
              resolve(blockId)
            } else {
              reject(new Error(`Block upload failed: ${xhr.status} ${xhr.statusText}`))
            }
          }
          
          xhr.onerror = () => reject(new Error('Network error during block upload'))
          xhr.ontimeout = () => reject(new Error('Block upload timeout'))
          
          xhr.open('PUT', blockUrl)
          xhr.setRequestHeader('Content-Type', 'application/octet-stream')
          xhr.setRequestHeader('x-ms-blob-type', 'BlockBlob')
          // Add rate limiting awareness headers
          xhr.setRequestHeader('X-Chunk-Upload', 'true')
          xhr.setRequestHeader('X-Expected-Chunks', blockCount.toString())
          xhr.setRequestHeader('X-Upload-Strategy', config.rateLimitStrategy || 'enhanced')
          xhr.send(chunk)
        })
        
        return blockId // Success, exit retry loop
        } catch (error) {
          // Enhanced error handling for rate limiting
          if (error.response?.status === 429) {
            const retryAfter = parseInt(error.response.headers['retry-after'] || '10')
            const rateLimitType = error.response.headers['x-ratelimit-type'] || 'unknown'
            console.warn(`Rate limited (${rateLimitType}), retrying after ${retryAfter}s (attempt ${attempt + 1}/${maxRetries + 1})`)
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000))
          } else if (attempt < maxRetries) {
            // Exponential backoff with jitter for other errors
            const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 1000, 10000)
            console.warn(`Block ${blockIndex} upload attempt ${attempt + 1} failed: ${error.message}. Retrying in ${delay}ms...`)
            await new Promise(resolve => setTimeout(resolve, delay))
          } else {
            console.error(`Block ${blockIndex} upload failed after ${maxRetries + 1} attempts: ${error.message}`)
            failedBlocks.add(blockIndex)
            throw error
          }
        }
    }
  }
  
  // Upload blocks with controlled concurrency
  const uploadTasks = []
  const semaphore = new Array(config.maxConcurrency).fill(null).map(() => Promise.resolve())
  let semaphoreIndex = 0
  
  for (let i = 0; i < blockCount; i++) {
    const currentIndex = semaphoreIndex
    semaphoreIndex = (semaphoreIndex + 1) % config.maxConcurrency
    
    const task = semaphore[currentIndex].then(() => uploadBlockWithRetry(i))
    semaphore[currentIndex] = task.catch(() => {}) // Prevent unhandled rejection
    uploadTasks.push(task)
  }
  
  // Wait for all uploads to complete
  const results = await Promise.allSettled(uploadTasks)
  
  // Check for failures
  const failures = results.filter(result => result.status === 'rejected')
  if (failures.length > 0) {
    const failureCount = failures.length
    const successCount = blockCount - failureCount
    console.error(`Upload failed: ${failureCount}/${blockCount} blocks failed to upload`)
    throw new Error(`Upload failed: ${failureCount}/${blockCount} blocks failed. ${successCount} blocks uploaded successfully.`)
  }
  
  // Commit the block list with retry
  const urlParts = uploadUrl.split('?')
  const baseUrl = urlParts[0]
  const sasToken = urlParts[1]
  const commitUrl = `${baseUrl}?comp=blocklist&${sasToken}`
  
  const blockListXml = `<?xml version="1.0" encoding="utf-8"?>
    <BlockList>
      ${blockIds.map(id => `<Latest>${id}</Latest>`).join('')}
    </BlockList>`
  
  // Commit the block list with retry (progress will naturally be at ~100% already)
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await new Promise((resolve, reject) => {
        const commitXhr = new XMLHttpRequest()
        commitXhr.timeout = 60000 // 1 minute timeout for commit
        
        commitXhr.onload = () => {
          if (commitXhr.status >= 200 && commitXhr.status < 300) {
            resolve()
          } else {
            reject(new Error(`Block list commit failed: ${commitXhr.status} ${commitXhr.statusText}`))
          }
        }
        
        commitXhr.onerror = () => reject(new Error('Network error during commit'))
        commitXhr.ontimeout = () => reject(new Error('Commit timeout'))
        
        commitXhr.open('PUT', commitUrl)
        commitXhr.setRequestHeader('Content-Type', 'application/xml')
        commitXhr.send(blockListXml)
      })
      
      console.log(`Successfully uploaded file using ${blockCount} parallel blocks`)
      // Ensure progress reaches 100% for the upload portion
      if (activeUploadIdRef.current === uploadId) {
        onProgress(1.0)
      }
      return // Success, exit retry loop
      
    } catch (error) {
      if (attempt < 2) {
        const delay = 1000 * Math.pow(2, attempt)
        console.warn(`Commit attempt ${attempt + 1} failed: ${error.message}. Retrying in ${delay}ms...`)
        await new Promise(resolve => setTimeout(resolve, delay))
      } else {
        console.error(`Commit failed after 3 attempts: ${error.message}`)
        throw error
      }
    }
  }
}

// Simple upload for smaller files with smooth progress
const uploadFileSimple = async (file, uploadUrl, onProgress, uploadId, activeUploadIdRef) => {
  console.log(`Using simple upload for file: ${file.name} (${file.size} bytes)`)
  
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    
    // Track upload progress smoothly
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        // Provide smooth progress updates during upload (0 to 1)
        const progress = e.loaded / e.total
        console.log(`Simple upload progress: ${(progress * 100).toFixed(1)}% (${e.loaded}/${e.total} bytes)`)
        
        // Check if upload is still active before calling progress callback
        if (activeUploadIdRef.current === uploadId) {
          onProgress(progress)
        }
      }
    })
    
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        if (activeUploadIdRef.current === uploadId) {
          onProgress(1.0) // Ensure we reach 100% on completion
        }
        resolve()
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`))
      }
    }
    
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.ontimeout = () => reject(new Error('Upload timeout'))
    
    xhr.open('PUT', uploadUrl)
    xhr.setRequestHeader('x-ms-blob-type', 'BlockBlob')
    if (file.type) {
      xhr.setRequestHeader('Content-Type', file.type)
    }
    xhr.send(file)
  })
}

const VideoUpload = ({ onJobCreated }) => {
  const { account, canAffordJob, refreshAccount } = useAccount()
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [attenuationDb, setAttenuationDb] = useState(30) // Default attenuation level
  const [activeUploadId, setActiveUploadId] = useState(null) // Track active upload to prevent races
  const [estimatedCost, setEstimatedCost] = useState(null)
  const [isCalculatingCost, setIsCalculatingCost] = useState(false)
  const fileInputRef = useRef(null)
  const activeUploadRef = useRef(null) // More reliable tracking using ref

  // Load account when component mounts
  useEffect(() => {
    refreshAccount()
  }, [])

  // Debug effect to track activeUploadId changes
  useEffect(() => {
    console.log(`activeUploadId changed to: ${activeUploadId}`)
  }, [activeUploadId])

  // Cleanup effect to handle component unmounting during active uploads
  useEffect(() => {
    return () => {
      // Component is unmounting, cancel any active uploads
      // Use a ref to get the current activeUploadId value
      setActiveUploadId(current => {
        if (current || activeUploadRef.current) {
          console.log('Component unmounting, cancelling active upload:', current || activeUploadRef.current)
        }
        return null
      })
      activeUploadRef.current = null
    }
  }, []) // Empty dependency array - only run on mount/unmount
  // Function to calculate cost based on file size (for UX only)
  const calculateCostForVideo = async (file) => {
    try {
      setIsCalculatingCost(true)
      
      // Calculate estimated cost based on file size for UX only
      // Note: Actual cost will be calculated securely on backend
      const estimatedCostCents = calculateJobCost(file.size)
      
      setEstimatedCost(estimatedCostCents)
      setIsCalculatingCost(false)
      
      return { fileSizeBytes: file.size, cost: estimatedCostCents }
    } catch (error) {
      console.error('Error calculating cost:', error)
      setIsCalculatingCost(false)
      return null
    }
  }

  const handleFileSelect = async (file) => {
    // Accept both audio and video types
    if (file && (file.type.startsWith('video/') || file.type.startsWith('audio/'))) {
      setSelectedFile(file)
      setEstimatedCost(null)
      
      // Calculate cost for the selected video
      await calculateCostForVideo(file)
    } else {
      alert('Please select a valid audio or video file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const removeSelectedFile = () => {
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleUpload = async () => {
    if (!selectedFile || isUploading || !estimatedCost) return

    // Check if user can afford the job
    if (!canAffordJob(estimatedCost)) {
      alert(`Insufficient account balance. You need $${(estimatedCost / 100).toFixed(2)} but only have $${((account?.balance || 0) / 100).toFixed(2)}.`)
      return
    }

    // Generate unique upload ID to prevent race conditions
    const uploadId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    console.log(`Starting new upload with ID: ${uploadId}`)
    
    // Check if there's already an active upload
    if (activeUploadId) {
      console.warn('Upload already in progress, ignoring new upload request')
      console.warn(`Current activeUploadId: ${activeUploadId}`)
      return
    }

    console.log(`Setting activeUploadId to: ${uploadId}`)
    setActiveUploadId(uploadId)
    activeUploadRef.current = uploadId
    
    // Small delay to ensure state is set before proceeding
    await new Promise(resolve => setTimeout(resolve, 10))
    setIsUploading(true)
    setUploadProgress(0)

    let uploadUrl = null
    let blobName = null
    let cleanup = null // Function to clean up resources

    try {
      // First, get the SAS upload URL from our API
      setUploadProgress(2) // Initial progress for getting upload URL
      
      // Get upload configuration for rate limiting awareness
      const uploadConfig = getUploadConfig(selectedFile.size)
      
      const uploadUrlResponse = await axios.post('/api/upload-file', {
        fileName: selectedFile.name,
        fileSize: selectedFile.size
      }, {
        headers: {
          'Content-Type': 'application/json',
          'X-Upload-Strategy': uploadConfig.rateLimitStrategy,
          'X-Expected-Chunks': uploadConfig.useParallel ? Math.ceil(selectedFile.size / uploadConfig.chunkSize).toString() : '1',
          'X-Parallel-Upload': uploadConfig.useParallel.toString()
        }
      })

      if (!uploadUrlResponse.data.success) {
        throw new Error(uploadUrlResponse.data.error || 'Failed to get upload URL')
      }

      // Extract variables from response
      uploadUrl = uploadUrlResponse.data.uploadUrl
      blobName = uploadUrlResponse.data.blobName
      
      setUploadProgress(5) // URL obtained, ready to start upload
      
      // Set up cleanup function for this specific upload
        cleanup = async () => {
          if (blobName) {
            try {
              await axios.delete(`/api/cleanup-blob`, {
                data: { blobName: blobName },
                headers: { 'Content-Type': 'application/json' }
              })
              console.log(`Cleaned up blob: ${blobName}`)
            } catch (cleanupError) {
              console.warn('Failed to cleanup blob:', cleanupError)
            }
          }
        }
      
      // Check if this upload is still the active one (prevent race conditions)
      // Only abort if BOTH state and ref indicate this upload is no longer active
      const isStateInvalid = activeUploadId !== null && activeUploadId !== uploadId
      const isRefInvalid = activeUploadRef.current !== null && activeUploadRef.current !== uploadId
      
      if (isStateInvalid && isRefInvalid) {
        console.warn('Upload was superseded by newer upload, aborting')
        console.warn(`Current activeUploadId: ${activeUploadId}, activeUploadRef: ${activeUploadRef.current}, this uploadId: ${uploadId}`)
        await cleanup()
        return
      } else if (activeUploadId !== uploadId || activeUploadRef.current !== uploadId) {
        console.log(`Race condition check: activeUploadId: ${activeUploadId}, activeUploadRef: ${activeUploadRef.current}, uploadId: ${uploadId} - continuing upload`)
      }
      
      // Upload directly to Azure Blob Storage
      // File upload takes 5% to 90% of progress bar for smooth visual feedback
      console.log(`Starting upload for file size: ${selectedFile.size} bytes`)
      console.log(`Upload config:`, getUploadConfig(selectedFile.size))
      
      await uploadFileInParallel(selectedFile, uploadUrl, (progress) => {
          // Check if this upload is still active before updating progress
          if (activeUploadRef.current === uploadId) {
            // Map file upload progress (0-1) to progress bar range (5-90%)
            const uploadProgressPercent = 5 + Math.round(progress * 85)
            setUploadProgress(uploadProgressPercent)
          }
        }, uploadId, activeUploadRef)

      // Final race condition check before job creation
      // Only abort if BOTH state and ref indicate this upload is no longer active
      const isFinalStateInvalid = activeUploadId !== null && activeUploadId !== uploadId
      const isFinalRefInvalid = activeUploadRef.current !== null && activeUploadRef.current !== uploadId
      
      if (isFinalStateInvalid && isFinalRefInvalid) {
        console.warn('Upload was superseded during file transfer, aborting job creation')
        console.warn(`Current activeUploadId: ${activeUploadId}, activeUploadRef: ${activeUploadRef.current}, this uploadId: ${uploadId}`)
        await cleanup()
        return
      } else if (activeUploadId !== uploadId || activeUploadRef.current !== uploadId) {
        console.log(`Upload still active after file transfer - activeUploadId: ${activeUploadId}, activeUploadRef: ${activeUploadRef.current}, uploadId: ${uploadId}`)
      }

      // File upload complete, preparing job creation
      setUploadProgress(92) // File uploaded, preparing job creation

      // Step 2: Create processing job using the blob info
      const jobData = {
        fileName: blobName,
        processingType: 'denoise',
        attenuationDb: attenuationDb
      }

      setUploadProgress(95) // Submitting job
      const jobResponse = await axios.post('/api/enqueue-job', jobData)
      setUploadProgress(98) // Job submitted
      
      console.log('Job response:', jobResponse.data)
      
      if (jobResponse.data.success || jobResponse.data.id) {
        console.log('Job created successfully, clearing selected file')
        const job = {
          id: jobResponse.data.id || jobResponse.data.jobId,
          filename: selectedFile.name,
          status: 'queued',
          progress: 0,
          createdAt: new Date().toISOString(),
          fileSize: selectedFile.size
        }

        onJobCreated(job)
        setSelectedFile(null)
        setEstimatedCost(null)
        
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        
        // Refresh account balance after successful job creation
        refreshAccount()
        
        setUploadProgress(100)
        
        alert('File uploaded successfully! Check the Processing Jobs tab to monitor progress.')
      } else {
        console.error('Job creation failed:', jobResponse.data)
        throw new Error(jobResponse.data.error || 'Job creation failed')
      }
    } catch (error) {
      console.error('Upload failed:', error)
      
      // Only clean up if this is still the active upload
      if (activeUploadId === uploadId && cleanup) {
        await cleanup()
      }
      
      // Only show alert if this is still the active upload (prevent stale error messages)
      if (activeUploadId === uploadId) {
        alert('Upload failed: ' + (error.response?.data?.error || error.message))
        setUploadProgress(0) // Reset on error
      }
    } finally {
      // Only reset state if this upload is still the active upload (using ref as primary check)
      if (activeUploadRef.current === uploadId) {
        console.log(`Upload ${uploadId} completed, resetting activeUploadId`)
        setIsUploading(false)
        setActiveUploadId(null)
        activeUploadRef.current = null
      } else {
        console.log(`Upload ${uploadId} not resetting state (activeUploadId: ${activeUploadId}, activeUploadRef: ${activeUploadRef.current})`)
      }
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="video-upload">
      <h2>Upload Audio or Video for Cleaning</h2>
      
      <div 
        className={`upload-zone ${isDragging ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload size={48} className="upload-icon" />
        <h3>Drop your media file here</h3>
        <p>or click to browse files</p>
        <p className="upload-note">Supported video: MP4, AVI, MOV, MKV, WEBM • Audio: MP3, WAV, M4A, AAC, FLAC, OGG, OPUS</p>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </div>

      {selectedFile && (
        <div className="selected-file">
          <div className="file-info">
            <FileVideo size={24} />
            <div className="file-details">
              <h4>{selectedFile.name}</h4>
              <p>{formatFileSize(selectedFile.size)}</p>
            </div>
            <button 
              onClick={removeSelectedFile}
              className="remove-file-btn"
              disabled={isUploading}
            >
              <X size={20} />
            </button>
          </div>

          {/* Cost and Duration Information */}
          <div className="cost-info">
            {isCalculatingCost ? (
              <div className="calculating-cost">
                <p>Calculating cost...</p>
              </div>
            ) : (
              <>
                {selectedFile && (
                  <div className="file-info">
                    <Clock size={16} />
                    <span>File Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                )}
                {estimatedCost && (
                  <div className="cost-display">
                    <DollarSign size={16} />
                    <span>Estimated Cost: ${(estimatedCost / 100).toFixed(2)} (final cost calculated securely on backend)</span>
                  </div>
                )}
                {account && (
                  <div className="balance-info">
                    <span>Account Balance: ${((account.balance || 0) / 100).toFixed(2)}</span>
                  </div>
                )}
                {estimatedCost && account && !canAffordJob(estimatedCost) && (
                  <div className="insufficient-funds">
                    <p>⚠️ Insufficient funds. Please add money to your account.</p>
                  </div>
                )}
              </>
            )}
          </div>
          
          {/* Attenuation Control */}
          <div className="attenuation-control">
            <label htmlFor="attenuation-slider" className="attenuation-label">
              Noise Reduction Strength: {attenuationDb} dB
            </label>
            <input
              id="attenuation-slider"
              type="range"
              min="10"
              max="50"
              value={attenuationDb}
              onChange={(e) => setAttenuationDb(parseInt(e.target.value))}
              className="attenuation-slider"
              disabled={isUploading}
            />
            <div className="attenuation-hints">
              <span className="hint-low">Gentle (10 dB)</span>
              <span className="hint-high">Aggressive (50 dB)</span>
            </div>
            <p className="attenuation-description">
              Higher values remove more noise but may affect audio quality. 
              Start with 30 dB for most videos.
            </p>
          </div>
          
          {isUploading && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p>Uploading... {uploadProgress}%</p>
            </div>
          )}
          
          <button 
            onClick={handleUpload}
            className="btn btn-primary upload-btn"
            disabled={
              isUploading || 
              activeUploadId !== null || 
              !estimatedCost || 
              !canAffordJob(estimatedCost) ||
              isCalculatingCost
            }
            title={
              activeUploadId ? "Upload already in progress" :
              !estimatedCost ? "Calculating cost..." :
              !canAffordJob(estimatedCost) ? "Insufficient account balance" :
              "Start processing this video"
            }
          >
            {isUploading ? 'Uploading...' : 
             isCalculatingCost ? 'Calculating Cost...' :
             'Start Audio Cleaning'}
          </button>
        </div>
      )}
    </div>
  )
}

export default VideoUpload
