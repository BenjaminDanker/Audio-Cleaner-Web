import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check if user is already authenticated using Static Web Apps
    checkAuthStatus()
    
    // Handle auth callback completion
    if (window.location.pathname === '/.auth/complete') {
      // Wait a moment for auth to process, then redirect
      setTimeout(() => {
        window.location.href = '/dashboard'
      }, 1000)
    }
  }, [])

  const checkAuthStatus = async () => {
    try {
      // Use Static Web Apps built-in authentication
      const response = await axios.get('/.auth/me', {
        timeout: 5000 // 5 second timeout
      })
      
      if (response.data.clientPrincipal && response.data.clientPrincipal.userId) {
        const principal = response.data.clientPrincipal
        setUser({
          id: principal.userId,
          email: principal.userDetails,
          name: principal.userDetails || principal.userRoles?.[0] || 'User'
        })
      } else {
        // No authenticated user
        setUser(null)
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      setUser(null)
    } finally {
      setLoading(false)
    }
  }

  const login = async () => {
    // SWA handles login automatically via staticwebapp.config.json routes
    // No manual redirect needed - just navigate to /login
    window.location.href = '/login'
    return { success: true }
  }

  const logout = () => {
    // SWA handles logout automatically via staticwebapp.config.json routes
    window.location.href = '/logout'
  }

  const value = {
    user,
    loading,
    login,
    logout
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
