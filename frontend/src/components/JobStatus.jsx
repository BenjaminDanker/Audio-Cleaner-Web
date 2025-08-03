import React, { useState, useEffect } from 'react'
import { Clock, CheckCircle, XCircle, Download, RefreshCw, Trash2 } from 'lucide-react'
import axios from 'axios'
import './JobStatus.css'

const JobStatus = ({ job, onUpdate, onDelete }) => {
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now())

  useEffect(() => {
    // Auto-refresh job status for active jobs only
    if (job.status === 'processing' || job.status === 'queued') {
      const interval = setInterval(() => {
        refreshJobStatus()
      }, 5000) // Check every 5 seconds

      return () => clearInterval(interval)
    }
    // Clear any existing intervals when job is completed/failed
  }, [job.status, job.id]) // Add job.id to dependencies to ensure clean intervals

  const refreshJobStatus = async () => {
    if (isRefreshing) return
    
    // Prevent too frequent updates (minimum 2 seconds between updates)
    const now = Date.now()
    if (now - lastUpdateTime < 2000) return
    
    setIsRefreshing(true)
    try {
      const response = await axios.get(`/api/job-status?jobId=${job.id}`)
      // The API returns the job data directly, not wrapped in a success object
      if (response.data && response.data.id) {
        // Extract only the fields we want to update to avoid conflicts
        const updates = {
          status: response.data.status,
          progress: response.data.progress || 0,
          message: response.data.message,
          updatedAt: response.data.updatedAt,
          completedAt: response.data.completedAt,
          downloadUrl: response.data.downloadUrl
        }
        
        // Only update if there are actual changes to prevent unnecessary re-renders
        const hasChanges = Object.keys(updates).some(key => 
          job[key] !== updates[key]
        )
        
        if (hasChanges) {
          onUpdate(updates)
          setLastUpdateTime(now)
        }
      }
    } catch (error) {
      console.error('Failed to refresh job status:', error)
    } finally {
      setIsRefreshing(false)
    }
  }

  const getStatusIcon = () => {
    switch (job.status) {
      case 'queued':
        return <Clock size={20} className="status-icon queued" />
      case 'processing':
        return <RefreshCw size={20} className="status-icon processing spinning" />
      case 'completed':
        return <CheckCircle size={20} className="status-icon completed" />
      case 'failed':
        return <XCircle size={20} className="status-icon failed" />
      default:
        return <Clock size={20} className="status-icon" />
    }
  }

  const getStatusText = () => {
    switch (job.status) {
      case 'queued':
        return 'Queued for processing'
      case 'processing':
        return 'Processing audio...'
      case 'completed':
        return 'Completed'
      case 'failed':
        return 'Processing failed'
      default:
        return 'Unknown status'
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString()
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getDownloadUrl = () => {
    const downloadUrl = job.downloadUrl
    if (!downloadUrl) return null
    
    // Handle local development URLs
    if (downloadUrl.startsWith('local://downloads/')) {
      const filename = downloadUrl.replace('local://downloads/', '')
      return `/api/download-file/${filename}`
    }
    
    // For production Azure Blob Storage URLs, extract filename and use download API
    if (downloadUrl.startsWith('https://')) {
      // Extract filename from URL like: https://storageaccount.blob.core.windows.net/container/filename.mp4
      const urlParts = downloadUrl.split('/')
      const filename = urlParts[urlParts.length - 1]
      return `/api/download-file/${filename}`
    }
    
    // If it's just a filename, construct the download API URL
    return `/api/download-file/${downloadUrl}`
  }

  const hasDownloadUrl = () => {
    return !!job.downloadUrl
  }

  // Parallel download helper for large files
  const downloadInParallel = async (downloadEndpoint, filename) => {
    setIsDownloading(true)
    setDownloadProgress(0)

    try {
      // Step 1: Get SAS URL from download endpoint
      console.log('Getting SAS URL from:', downloadEndpoint)
      const response = await fetch(downloadEndpoint, { method: 'GET' })
      
      if (!response.ok) {
        throw new Error(`Failed to get SAS URL: ${response.status} ${response.statusText}`)
      }
      
      const data = await response.json()
      const sasUrl = data.sasUrl
      const totalSize = data.contentLength
      
      if (!sasUrl) {
        throw new Error('No SAS URL received from API')
      }
      
      if (!totalSize || totalSize <= 0) {
        throw new Error('Could not determine file size or file is empty')
      }
      
      console.log('SAS URL received, downloading directly from blob storage')
      console.log('File size:', totalSize, 'bytes')

      // Step 2: Setup chunk size and ranges
      const chunkSize = 4 * 1024 * 1024 // 4 MB
      const chunkCount = Math.ceil(totalSize / chunkSize)
      const ranges = Array.from({ length: chunkCount }, (_, i) => {
        const start = i * chunkSize
        const end = Math.min(totalSize - 1, (i + 1) * chunkSize - 1)
        return { index: i, range: `bytes=${start}-${end}` }
      })

      // Step 3: Track progress with completed chunks
      let completedChunks = 0
      const updateProgress = () => {
        completedChunks++
        const progress = Math.floor((completedChunks / chunkCount) * 100)
        setDownloadProgress(progress)
      }

      // Step 4: Fetch all chunks in parallel using the SAS URL
      const chunks = await Promise.all(
        ranges.map(async ({ index, range }) => {
          try {
            const res = await fetch(sasUrl, {
              headers: { Range: range }
            })
            
            if (!res.ok) {
              throw new Error(`Failed to fetch chunk ${index}: ${res.status}`)
            }
            
            const buffer = await res.arrayBuffer()
            updateProgress()
            return { index, buffer }
          } catch (error) {
            console.error(`Error downloading chunk ${index}:`, error)
            throw error
          }
        })
      )

      // Step 5: Sort and stitch
      chunks.sort((a, b) => a.index - b.index)
      const blob = new Blob(chunks.map(c => c.buffer), { 
        type: data.contentType || 'application/octet-stream' 
      })

      // Step 6: Trigger download
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      
      console.log('Download completed successfully')
    } catch (err) {
      console.error('Parallel download failed:', err)
      // Fallback: try to redirect to the original SAS URL for direct download
      try {
        const response = await fetch(downloadEndpoint, { method: 'GET' })
        if (response.ok) {
          const data = await response.json()
          if (data.sasUrl) {
            window.open(data.sasUrl, '_blank')
            return
          }
        }
      } catch (fallbackError) {
        console.error('Fallback download also failed:', fallbackError)
      }
      
      alert(`Download failed: ${err.message}. Please try again or contact support.`)
    } finally {
      setIsDownloading(false)
      setDownloadProgress(0)
    }
  };

  return (
    <div className={`job-status ${job.status}`}>
      <div className="job-header">
        <div className="job-info">
          {getStatusIcon()}
          <div className="job-details">
            <h3>{job.filename}</h3>
            <p className="job-meta">
              {formatFileSize(job.fileSize)} • Created {formatDate(job.createdAt)}
            </p>
          </div>
        </div>
        
        <div className="job-actions">
          <button 
            onClick={refreshJobStatus}
            className="refresh-btn"
            disabled={isRefreshing}
            title="Refresh status"
          >
            <RefreshCw size={16} className={isRefreshing ? 'spinning' : ''} />
          </button>
          
          <button 
            onClick={onDelete}
            className="delete-btn"
            title="Delete this job"
          >
            <Trash2 size={16} />
          </button>
          
          {job.status === 'completed' && hasDownloadUrl() && (
            <button
              onClick={() => downloadInParallel(getDownloadUrl(), job.filename)}
              className="btn btn-sm btn-primary"
              title="Download"
              disabled={isDownloading}
            >
              <Download size={16} />
              Download
            </button>
          )}
        </div>
      </div>

      <div className="job-status-text">
        <span>{getStatusText()}</span>
      </div>

      {(job.status === 'processing' || job.status === 'queued') && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-bar-fill"
              style={{ width: `${job.progress || 0}%` }}
            ></div>
          </div>
          <span className="progress-text">{job.progress || 0}%</span>
        </div>
      )}

      {isDownloading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{ width: `${downloadProgress}%` }}
            />
          </div>
          <span className="progress-text">Downloading {downloadProgress}%</span>
        </div>
      )}

      {job.status === 'failed' && job.error && (
        <div className="error-message">
          <p><strong>Error:</strong> {job.error}</p>
        </div>
      )}

      {job.status === 'completed' && (
        <div className="completion-message">
          {hasDownloadUrl() ? (
            <p>Your audio has been cleaned and is ready for download!</p>
          ) : (
            <p>Processing completed, but download link is not available. Please contact support.</p>
          )}
        </div>
      )}
    </div>
  )
}

export default JobStatus
