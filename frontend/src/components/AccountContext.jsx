import React, { createContext, useContext, useState } from 'react'
import axios from 'axios'

const AccountContext = createContext()

export const useAccount = () => {
  const context = useContext(AccountContext)
  if (!context) {
    throw new Error('useAccount must be used within an AccountProvider')
  }
  return context
}

export const AccountProvider = ({ children }) => {
  const [account, setAccount] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchAccount = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/get-account-data')
      setAccount(response.data.account)
    } catch (error) {
      console.error('Failed to fetch account:', error)
      // Set default account if none exists
      setAccount({
        id: 'default',
        userId: 'current-user',
        balance: 0,
        currency: 'usd',
        createdAt: new Date().toISOString()
      })
    } finally {
      setLoading(false)
    }
  }

  const loadAccount = async () => {
    if (!account && !loading) {
      await fetchAccount()
    }
  }

  const refreshAccount = async () => {
    await fetchAccount()
  }

  const canAffordJob = (estimatedCost) => {
    return account && account.balance >= estimatedCost
  }

  const getBalanceInDollars = () => {
    return account ? account.balance / 100 : 0
  }

  const value = {
    account,
    loading,
    loadAccount,
    refreshAccount,
    canAffordJob,
    getBalanceInDollars
  }

  return (
    <AccountContext.Provider value={value}>
      {children}
    </AccountContext.Provider>
  )
}
