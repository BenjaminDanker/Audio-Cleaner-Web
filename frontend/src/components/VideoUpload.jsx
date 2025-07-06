import React, { useState, useRef } from 'react'
import { Upload, FileVideo, X } from 'lucide-react'
import axios from 'axios'
import './VideoUpload.css'

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
        // Production: Upload directly to Azure Blob Storage using the SAS URL
        const config = {
          onUploadProgress: (progressEvent) => {
            const progress = 20 + Math.round((progressEvent.loaded / progressEvent.total) * 60)
            setUploadProgress(progress)
          },
          headers: {
            'x-ms-blob-type': 'BlockBlob'
          }
        }
        
        // Use XMLHttpRequest directly to avoid Axios CORS complications
        const xhr = new XMLHttpRequest()
        
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const progress = 20 + Math.round((e.loaded / e.total) * 60)
            setUploadProgress(progress)
          }
        })
        
        await new Promise((resolve, reject) => {
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve(xhr.response)
            } else {
              reject(new Error(`Upload failed with status ${xhr.status}: ${xhr.statusText}`))
            }
          }
          
          xhr.onerror = () => reject(new Error('Network error during upload'))
          xhr.ontimeout = () => reject(new Error('Upload timeout'))
          
          xhr.open('PUT', uploadUrl)
          xhr.setRequestHeader('x-ms-blob-type', 'BlockBlob')
          
          // Set content type if available
          if (selectedFile.type) {
            xhr.setRequestHeader('Content-Type', selectedFile.type)
          }
          
          xhr.send(selectedFile)
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
