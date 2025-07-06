import React, { useState, useEffect } from 'react'
import { Clock, CheckCircle, XCircle, Download, RefreshCw, Trash2 } from 'lucide-react'
import axios from 'axios'
import './JobStatus.css'

const JobStatus = ({ job, onUpdate, onDelete }) => {
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    // Auto-refresh job status for active jobs
    if (job.status === 'processing' || job.status === 'queued') {
      const interval = setInterval(() => {
        refreshJobStatus()
      }, 5000) // Check every 5 seconds

      return () => clearInterval(interval)
    }
  }, [job.status])

  const refreshJobStatus = async () => {
    if (isRefreshing) return
    
    setIsRefreshing(true)
    try {
      const response = await axios.get(`/api/job-status?jobId=${job.id}`)
      // The API returns the job data directly, not wrapped in a success object
      if (response.data && response.data.id) {
        onUpdate(response.data)
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
    // Check for different possible field names for the download URL
    const downloadUrl = job.downloadUrl || job.output_blob_url || job.outputBlobUrl
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
    return !!(job.downloadUrl || job.output_blob_url || job.outputBlobUrl)
  }

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
            <a 
              href={getDownloadUrl()} 
              download
              className="btn btn-sm btn-primary"
            >
              <Download size={16} />
              Download
            </a>
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
