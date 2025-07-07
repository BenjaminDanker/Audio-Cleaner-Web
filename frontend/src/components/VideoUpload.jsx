import React, { useState, useRef } from 'react'
import { Upload, FileVideo, X } from 'lucide-react'
import axios from 'axios'
import './VideoUpload.css'

// Utility function for parallel block uploads with Azure best practices
const uploadFileInParallel = async (file, uploadUrl, onProgress) => {
  const fileSize = file.size
  const blockSize = 4 * 1024 * 1024 // 4MB blocks (optimal for Azure)
  const maxConcurrency = 6 // Maximum parallel uploads
  
  // For files smaller than 32MB, use simple upload
  if (fileSize < 32 * 1024 * 1024) {
    return uploadFileSimple(file, uploadUrl, onProgress)
  }
  
  const blockCount = Math.ceil(fileSize / blockSize)
  const blockIds = []
  
  // Generate block IDs with proper encoding
  for (let i = 0; i < blockCount; i++) {
    // Use URL-safe base64 encoding
    const blockId = btoa(`block-${i.toString().padStart(6, '0')}`).replace(/[+/]/g, (m) => m === '+' ? '-' : '_').replace(/=+$/, '')
    blockIds.push(blockId)
  }
  
  let uploadedBlocks = 0
  const failedBlocks = new Set()
  
  // Enhanced block upload with retry logic
  const uploadBlockWithRetry = async (blockIndex, maxRetries = 3) => {
    const blockId = blockIds[blockIndex]
    const start = blockIndex * blockSize
    const end = Math.min(start + blockSize, fileSize)
    const chunk = file.slice(start, end)
    
    // Parse the upload URL to get base URL and SAS token
    const urlParts = uploadUrl.split('?')
    const baseUrl = urlParts[0]
    const sasToken = urlParts[1]
    
    const blockUrl = `${baseUrl}?comp=block&blockid=${encodeURIComponent(blockId)}&${sasToken}`
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          
          // Set timeout (2 minutes per block)
          xhr.timeout = 120000
          
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              uploadedBlocks++
              onProgress(uploadedBlocks / blockCount)
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
          xhr.send(chunk)
        })
        
        return blockId // Success, exit retry loop
        
      } catch (error) {
        if (attempt < maxRetries) {
          // Exponential backoff with jitter
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
  const semaphore = new Array(maxConcurrency).fill(null).map(() => Promise.resolve())
  let semaphoreIndex = 0
  
  for (let i = 0; i < blockCount; i++) {
    const currentIndex = semaphoreIndex
    semaphoreIndex = (semaphoreIndex + 1) % maxConcurrency
    
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
  
  // Retry commit operation
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

// Simple upload for smaller files
const uploadFileSimple = async (file, uploadUrl, onProgress) => {
  const xhr = new XMLHttpRequest()
  
  xhr.upload.addEventListener('progress', (e) => {
    if (e.lengthComputable) {
      onProgress(e.loaded / e.total)
    }
  })
  
  return new Promise((resolve, reject) => {
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
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
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [attenuationDb, setAttenuationDb] = useState(30) // Default attenuation level
  const fileInputRef = useRef(null)

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file)
    } else {
      alert('Please select a valid video file')
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
    if (!selectedFile) return

    setIsUploading(true)
    setUploadProgress(0)

    let uploadUrl = null
    let blobName = null
    let fileUrl = null

    try {
      // First, get the SAS upload URL from our API
      setUploadProgress(10)
      
      const uploadUrlResponse = await axios.post('/api/upload-file', {
        fileName: selectedFile.name,
        fileSize: selectedFile.size
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!uploadUrlResponse.data.success) {
        throw new Error(uploadUrlResponse.data.error || 'Failed to get upload URL')
      }

      // Extract variables from response
      uploadUrl = uploadUrlResponse.data.uploadUrl
      fileUrl = uploadUrlResponse.data.fileUrl
      blobName = uploadUrlResponse.data.blobName
      
      setUploadProgress(20)

      // Check if we're in local development
      const isLocalDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      
      if (isLocalDev) {
        // Local development: Simulate upload
        setUploadProgress(80)
      } else {
        // Production: Upload directly to Azure Blob Storage using parallel block uploads
        await uploadFileInParallel(selectedFile, uploadUrl, (progress) => {
          setUploadProgress(20 + Math.round(progress * 60))
        })
      }

      setUploadProgress(85)

      // Step 2: Create processing job using the blob info
      const jobData = {
        fileName: blobName,
        fileUrl: fileUrl,
        processingType: 'denoise',
        attenuationDb: attenuationDb
      }

      const jobResponse = await axios.post('/api/enqueue-job', jobData)
      setUploadProgress(95)
      
      if (jobResponse.data.success || jobResponse.data.id) {
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
        
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        
        setUploadProgress(100)
        
        alert('File uploaded successfully! Check the Processing Jobs tab to monitor progress.')
      } else {
        throw new Error(jobResponse.data.error || 'Job creation failed')
      }
    } catch (error) {
      console.error('Upload failed:', error)
      
      // Clean up any partial blob upload if it exists
      if (uploadUrl && blobName) {
        try {
          await axios.delete(uploadUrl)
          console.log('Cleaned up partial blob upload')
        } catch (cleanupError) {
          console.warn('Failed to cleanup partial blob:', cleanupError)
        }
      }
      
      alert('Upload failed: ' + (error.response?.data?.error || error.message))
      setUploadProgress(0) // Reset on error
    } finally {
      setIsUploading(false)
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
      <h2>Upload Video for Audio Cleaning</h2>
      
      <div 
        className={`upload-zone ${isDragging ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload size={48} className="upload-icon" />
        <h3>Drop your video file here</h3>
        <p>or click to browse files</p>
        <p className="upload-note">Supported formats: MP4, AVI, MOV, MKV</p>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
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
            disabled={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Start Audio Cleaning'}
          </button>
        </div>
      )}
    </div>
  )
}

export default VideoUpload
