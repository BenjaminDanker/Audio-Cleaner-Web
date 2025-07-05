import React from 'react'
import { useAuth } from './AuthContext'
import { Navigate } from 'react-router-dom'
import './Login.css'

const Login = () => {
  const { login, user } = useAuth()

  if (user) {
    return <Navigate to="/dashboard" replace />
  }

  const handleLogin = () => {
    login()
  }

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Audio Cleaner Pro</h1>
        <h2>Sign In to Your Account</h2>
        <p className="login-description">
          Sign in with your Azure Active Directory account to access the dashboard and start cleaning your audio files.
        </p>
        
        <button 
          onClick={handleLogin}
          className="btn btn-primary login-button"
        >
          Sign In with Azure AD
        </button>
        
        <div className="login-footer">
          <p>This app uses Azure Static Web Apps authentication</p>
        </div>
      </div>
    </div>
  )
}

export default Login
