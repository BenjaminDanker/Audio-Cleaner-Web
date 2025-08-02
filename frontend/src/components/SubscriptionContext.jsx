import React, { createContext, useContext, useState } from 'react'
import axios from 'axios'

const SubscriptionContext = createContext()

export const useSubscription = () => {
  const context = useContext(SubscriptionContext)
  if (!context) {
    throw new Error('useSubscription must be used within a SubscriptionProvider')
  }
  return context
}

export const SubscriptionProvider = ({ children }) => {
  const [subscription, setSubscription] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchSubscription = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/get-subscription')
      setSubscription(response.data)
    } catch (error) {
      console.error('Failed to fetch subscription:', error)
      // Optionally set an error state here
    } finally {
      setLoading(false)
    }
  }

  const loadSubscription = async () => {
    if (!subscription && !loading) {
      await fetchSubscription()
    }
  }

  const refreshSubscription = async () => {
    await fetchSubscription()
  }

  const value = {
    subscription,
    loading,
    loadSubscription,
    refreshSubscription
  }

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  )
}
