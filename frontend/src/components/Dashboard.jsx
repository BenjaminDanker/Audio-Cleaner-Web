import React, { useState, useEffect } from 'react'
import { useAuth } from './AuthContext'
import VideoUpload from './VideoUpload'
import JobStatus from './JobStatus'
import SubscriptionInfo from './SubscriptionInfo'
import './Dashboard.css'

const Dashboard = () => {
  const { user, subscription } = useAuth()
  const [activeTab, setActiveTab] = useState('upload')
  const [jobs, setJobs] = useState([])

  useEffect(() => {
    // Load any existing jobs from localStorage or API
    const savedJobs = localStorage.getItem('audio_cleaner_jobs')
    if (savedJobs) {
      setJobs(JSON.parse(savedJobs))
    }
  }, [])

  const addJob = (job) => {
    const newJobs = [...jobs, job]
    setJobs(newJobs)
    localStorage.setItem('audio_cleaner_jobs', JSON.stringify(newJobs))
  }

  const updateJob = (jobId, updates) => {
    const updatedJobs = jobs.map(job => 
      job.id === jobId ? { ...job, ...updates } : job
    )
    setJobs(updatedJobs)
    localStorage.setItem('audio_cleaner_jobs', JSON.stringify(updatedJobs))
  }

  const deleteJob = async (jobId) => {
    if (!window.confirm('Are you sure you want to delete this job? This will permanently delete the job record and associated files.')) {
      return
    }

    try {
      const response = await fetch(`/api/clear-jobs?jobId=${jobId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (result.success) {
        const updatedJobs = jobs.filter(job => job.id !== jobId)
        setJobs(updatedJobs)
        localStorage.setItem('audio_cleaner_jobs', JSON.stringify(updatedJobs))
        alert('Job deleted successfully')
      } else {
        alert(`Error deleting job: ${result.error}`)
      }
    } catch (error) {
      console.error('Error deleting job:', error)
      alert('Error deleting job. Please try again.')
    }
  }

  const clearAllJobs = async () => {
    if (!window.confirm('Are you sure you want to clear all jobs? This will permanently delete all job records.')) {
      return
    }

    try {
      const response = await fetch('/api/clear-jobs', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (result.success) {
        setJobs([])
        localStorage.removeItem('audio_cleaner_jobs')
        alert(`Successfully cleared ${result.deletedCount} jobs`)
      } else {
        alert(`Error clearing jobs: ${result.error}`)
      }
    } catch (error) {
      console.error('Error clearing jobs:', error)
      alert('Error clearing jobs. Please try again.')
    }
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Welcome back, {user?.name || 'User'}!</h1>
        <p>Clean your audio files with AI-powered noise reduction</p>
      </div>

      <div className="dashboard-tabs">
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Video
        </button>
        <button 
          className={`tab ${activeTab === 'jobs' ? 'active' : ''}`}
          onClick={() => setActiveTab('jobs')}
        >
          Processing Jobs ({jobs.length})
        </button>
        <button 
          className={`tab ${activeTab === 'subscription' ? 'active' : ''}`}
          onClick={() => setActiveTab('subscription')}
        >
          Subscription
        </button>
      </div>

      <div className="dashboard-content">
        {activeTab === 'upload' && (
          <VideoUpload onJobCreated={addJob} />
        )}
        
        {activeTab === 'jobs' && (
          <div className="jobs-container">
            <div className="jobs-header">
              <h2>Processing Jobs</h2>
              {jobs.length > 0 && (
                <button 
                  className="clear-jobs-btn"
                  onClick={clearAllJobs}
                  title="Clear all your job records and associated files"
                >
                  Clear All Jobs
                </button>
              )}
            </div>
            {jobs.length === 0 ? (
              <p className="no-jobs">No jobs yet. Upload a video to get started!</p>
            ) : (
              <div className="jobs-list">
                {jobs.map(job => (
                  <JobStatus 
                    key={job.id} 
                    job={job} 
                    onUpdate={(updates) => updateJob(job.id, updates)}
                    onDelete={() => deleteJob(job.id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'subscription' && (
          <SubscriptionInfo subscription={subscription} />
        )}
      </div>
    </div>
  )
}

export default Dashboard
