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
      // Check if we're in development mode (Static Web Apps auth not available locally)
      const isDevelopment = window.location.hostname === 'localhost' || 
                           window.location.hostname === '127.0.0.1' ||
                           window.location.port === '5173'
      
      if (isDevelopment) {
        // For local development, simulate a logged-in user
        console.log('Development mode detected - simulating authenticated user')
        setUser({
          id: 'dev-user-123',
          email: 'dev@example.com',
          name: 'Development User'
        })
        await fetchSubscription()
      } else {
        // Use Static Web Apps built-in authentication in production
        try {
          const response = await axios.get('/.auth/me')
          
          if (response.data.clientPrincipal) {
            const principal = response.data.clientPrincipal
            setUser({
              id: principal.userId,
              email: principal.userDetails,
              name: principal.userDetails
            })
            await fetchSubscription()
          }
        } catch (authError) {
          // If auth endpoint fails, user is not authenticated
          console.log('User not authenticated')
          setUser(null)
        }
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      // In case of error, check if we're in development
      const isDevelopment = window.location.hostname === 'localhost' || 
                           window.location.hostname === '127.0.0.1' ||
                           window.location.port === '5173'
      if (isDevelopment) {
        setUser({
          id: 'dev-user-123',
          email: 'dev@example.com',
          name: 'Development User'
        })
      }
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
    // Check if we're in development mode
    const isDevelopment = window.location.hostname === 'localhost' || 
                         window.location.hostname === '127.0.0.1' ||
                         window.location.port === '5173'
    
    if (isDevelopment) {
      // For local development, just simulate login
      console.log('Development login - simulating authenticated user')
      setUser({
        id: 'dev-user-123',
        email: 'dev@example.com',
        name: 'Development User'
      })
      await fetchSubscription()
      return { success: true }
    } else {
      // For Static Web Apps, redirect to the built-in login
      // Remove the period to make it compatible with Azure Functions routing
      window.location.href = '/.auth/login/aad'
      return { success: true }
    }
  }

  const logout = () => {
    // Check if we're in development mode
    const isDevelopment = window.location.hostname === 'localhost' || 
                         window.location.hostname === '127.0.0.1' ||
                         window.location.port === '5173'
    
    if (isDevelopment) {
      // For local development, just clear user state
      console.log('Development logout - clearing user state')
      setUser(null)
      setSubscription(null)
    } else {
      // Use Static Web Apps built-in logout
      window.location.href = '/.auth/logout'
    }
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
