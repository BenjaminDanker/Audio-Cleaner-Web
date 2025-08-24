import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { useAuth } from './AuthContext'
import { useAccount } from './AccountContext'
import VideoUpload from './VideoUpload'
import JobStatus from './JobStatus'
import AccountBalance from './AccountBalance'
import ApiKeys from './ApiKeys'
import './Dashboard.css'

const Dashboard = () => {
  const { user } = useAuth()
  const { loadAccount, account, loading: accountLoading } = useAccount()
  const [activeTab, setActiveTab] = useState('upload')
  const [jobs, setJobs] = useState([])
  const [paymentMessage, setPaymentMessage] = useState(null)

  useEffect(() => {
    // Check for payment status in URL params
    const urlParams = new URLSearchParams(window.location.search)
    if (urlParams.get('payment_success') === 'true') {
      setPaymentMessage({ type: 'success', text: 'Payment successful! Your account has been credited.' })
      setActiveTab('account') // Switch to account tab to show updated balance
      // Clean up URL
      window.history.replaceState({}, document.title, '/dashboard')
    } else if (urlParams.get('payment_cancelled') === 'true') {
      setPaymentMessage({ type: 'error', text: 'Payment was cancelled.' })
      // Clean up URL
      window.history.replaceState({}, document.title, '/dashboard')
    }

    // Auto-hide payment message after 5 seconds
    if (paymentMessage) {
      const timer = setTimeout(() => setPaymentMessage(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [paymentMessage])

  useEffect(() => {
    // Load any existing jobs from localStorage or API
    const savedJobs = localStorage.getItem('audio_cleaner_jobs')
    if (savedJobs) {
      setJobs(JSON.parse(savedJobs))
    }
  }, [])

  // Prefetch account data once user is available so we can decide when to show the tab
  useEffect(() => {
    if (user && !account && !accountLoading) {
      loadAccount()
    }
  }, [user, account, accountLoading, loadAccount])

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
        <p>Clean audio from your audio or video files with AI-powered noise reduction</p>
      </div>

      {paymentMessage && (
        <div className={`payment-message ${paymentMessage.type}`}>
          {paymentMessage.text}
          <button onClick={() => setPaymentMessage(null)}>×</button>
        </div>
      )}

      <div className="dashboard-tabs">
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Media
        </button>
        <button 
          className={`tab ${activeTab === 'jobs' ? 'active' : ''}`}
          onClick={() => setActiveTab('jobs')}
        >
          Processing Jobs ({jobs.length})
        </button>
        <button 
          className={`tab ${activeTab === 'account' ? 'active' : ''}`}
          onClick={() => setActiveTab('account')}
        >
          Account Balance
        </button>
        <button 
          className={`tab ${activeTab === 'keys' ? 'active' : ''}`}
          onClick={() => setActiveTab('keys')}
        >
          API Keys
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
        
        {activeTab === 'account' && (
          <AccountBalance />
        )}

        {activeTab === 'keys' && (
          <ApiKeys />
        )}
      </div>
    </div>
  )
}

export default Dashboard
