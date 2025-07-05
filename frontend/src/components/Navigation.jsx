import React from 'react'
import { useAuth } from './AuthContext'
import { LogOut, User } from 'lucide-react'
import './Navigation.css'

const Navigation = () => {
  const { user, logout } = useAuth()

  const handleLogout = () => {
    logout()
  }

  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <h2>Audio Cleaner Pro</h2>
        </div>
        
        <div className="nav-user">
          <div className="user-info">
            <User size={20} />
            <span>{user?.name || user?.email || 'User'}</span>
          </div>
          
          <button 
            onClick={handleLogout}
            className="logout-btn"
            title="Logout"
          >
            <LogOut size={20} />
            <span>Logout</span>
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navigation
