import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { useAuth } from './AuthContext'
import { useSubscription } from './SubscriptionContext'
import VideoUpload from './VideoUpload'
import JobStatus from './JobStatus'
import SubscriptionInfo from './SubscriptionInfo'
import './Dashboard.css'

const Dashboard = () => {
  const { user } = useAuth()
  const { loadSubscription } = useSubscription()
  const [activeTab, setActiveTab] = useState('upload')
  const [jobs, setJobs] = useState([])

  useEffect(() => {
    // Load any existing jobs from localStorage or API
    const savedJobs = localStorage.getItem('audio_cleaner_jobs')
    if (savedJobs) {
      setJobs(JSON.parse(savedJobs))
    }
  }, [])

  // Load subscription data when subscription tab is activated
  useEffect(() => {
    if (activeTab === 'subscription') {
      loadSubscription()
    }
  }, [activeTab, loadSubscription])

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
      // Always remove from localStorage first
      const updatedJobs = jobs.filter(job => job.id !== jobId)
      setJobs(updatedJobs)
      localStorage.setItem('audio_cleaner_jobs', JSON.stringify(updatedJobs))
      
      // Try to delete from API, but don't fail if it doesn't exist
      try {
        const response = await axios.delete(`/api/clear-jobs?jobId=${jobId}`)
        console.log('API delete result:', response.data)
      } catch (apiError) {
        console.warn('API delete failed (job may not exist in database):', apiError)
      }
      
      alert('Job deleted successfully')
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
      // Always clear localStorage first
      setJobs([])
      localStorage.removeItem('audio_cleaner_jobs')
      
      // Try to clear from API, but don't fail if it doesn't work
      try {
        const response = await axios.delete('/api/clear-jobs')
        console.log('API clear result:', response.data)
        alert(`Successfully cleared all jobs${response.data.deletedCount ? ` (${response.data.deletedCount} from database)` : ''}`)
      } catch (apiError) {
        console.warn('API clear failed:', apiError)
        alert('All jobs cleared from browser (database may be empty)')
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
          <SubscriptionInfo />
        )}
      </div>
    </div>
  )
}

export default Dashboard
