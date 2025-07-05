import React, { useState, useRef } from 'react'
import { Upload, FileVideo, X } from 'lucide-react'
import axios from 'axios'
import './VideoUpload.css'

const VideoUpload = ({ onJobCreated }) => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
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

    try {
      // Step 1: Upload file to Azure Blob Storage
      const formData = new FormData()
      formData.append('file', selectedFile)
      
      const uploadResponse = await axios.post('/api/upload-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded / progressEvent.total) * 80) // 80% for upload
          setUploadProgress(progress)
        }
      })

      if (!uploadResponse.data.success) {
        throw new Error(uploadResponse.data.error || 'File upload failed')
      }

      const { fileUrl, fileName } = uploadResponse.data
      setUploadProgress(85)

      // Step 2: Create processing job
      const jobData = {
        fileName: fileName,
        fileUrl: fileUrl,
        processingType: 'denoise'
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
