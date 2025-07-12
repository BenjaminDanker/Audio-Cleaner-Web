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
  const [subscription, setSubscription] = useState(null)

  useEffect(() => {
    // Check if user is already authenticated using Static Web Apps
    checkAuthStatus()
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
        await fetchSubscription()
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

  const fetchSubscription = async () => {
    try {
      const response = await axios.get('/api/get-subscription')
      setSubscription(response.data)
    } catch (error) {
      console.error('Failed to fetch subscription:', error)
    }
  }

  const login = async () => {
    // For Static Web Apps, redirect to the built-in login
    window.location.href = '/.auth/login/aad'
    return { success: true }
  }

  const logout = () => {
    // Use Static Web Apps built-in logout
    window.location.href = '/.auth/logout'
  }

  const value = {
    user,
    subscription,
    loading,
    login,
    logout,
    refreshSubscription: fetchSubscription
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
