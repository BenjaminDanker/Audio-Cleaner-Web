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
            <h2>Processing Jobs</h2>
            {jobs.length === 0 ? (
              <p className="no-jobs">No jobs yet. Upload a video to get started!</p>
            ) : (
              <div className="jobs-list">
                {jobs.map(job => (
                  <JobStatus 
                    key={job.id} 
                    job={job} 
                    onUpdate={(updates) => updateJob(job.id, updates)}
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
