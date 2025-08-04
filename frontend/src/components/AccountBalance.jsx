import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { CreditCard, DollarSign, Upload, History } from 'lucide-react'
import './AccountBalance.css'

const AccountBalance = () => {
  const [account, setAccount] = useState(null)
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(false)
  const [addAmount, setAddAmount] = useState('')

  useEffect(() => {
    fetchAccountData()
  }, [])

  const fetchAccountData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/get-account-data')
      setAccount(response.data.account)
      setTransactions(response.data.transactions)
    } catch (error) {
      console.error('Failed to fetch account data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleAddFunds = async () => {
    const amount = parseFloat(addAmount)
    if (!amount || amount <= 0) {
      alert('Please enter a valid amount')
      return
    }

    try {
      const response = await axios.post('/api/create-payment-session', {
        amount: amount * 100, // Convert to cents for Stripe
        currency: 'usd'
      })

      if (response.data.url) {
        window.location.href = response.data.url
      }
    } catch (error) {
      console.error('Error creating payment session:', error)
      alert('Failed to initiate payment. Please try again.')
    }
  }

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount / 100) // Convert cents to dollars
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading && !account) {
    return (
      <div className="account-loading">
        <p>Loading account information...</p>
      </div>
    )
  }

  return (
    <div className="account-balance">
      <div className="account-header">
        <h2>Account Balance</h2>
        <button onClick={fetchAccountData} className="btn btn-secondary">
          Refresh
        </button>
      </div>

      <div className="account-cards">
        <div className="account-card balance-card">
          <div className="card-header">
            <DollarSign size={24} />
            <h3>Current Balance</h3>
          </div>
          <div className="balance-info">
            <div className="balance-amount">
              {formatCurrency(account?.balance || 0)}
            </div>
            <div className="balance-status">
              Available for video processing
            </div>
          </div>
        </div>

        <div className="account-card add-funds-card">
          <div className="card-header">
            <CreditCard size={24} />
            <h3>Add Funds</h3>
          </div>
          <div className="add-funds-form">
            <div className="input-group">
              <span className="currency-symbol">$</span>
              <input
                type="number"
                value={addAmount}
                onChange={(e) => setAddAmount(e.target.value)}
                placeholder="0.00"
                min="1"
                step="0.01"
                className="amount-input"
              />
            </div>
            <button 
              onClick={handleAddFunds}
              className="btn btn-primary"
              disabled={!addAmount || parseFloat(addAmount) <= 0}
            >
              <Upload size={16} />
              Add Funds
            </button>
          </div>
        </div>

        <div className="account-card pricing-card">
          <div className="card-header">
            <Upload size={24} />
            <h3>Pricing Information</h3>
          </div>
          <div className="pricing-info">
            <div className="pricing-item">
              <span className="pricing-label">Video Processing:</span>
              <span className="pricing-value">$0.50 per minute</span>
            </div>
            <div className="pricing-note">
              Cost is calculated based on video duration before processing
            </div>
          </div>
        </div>
      </div>

      <div className="transactions-section">
        <div className="section-header">
          <div className="header-title">
            <History size={20} />
            <h3>Transaction History</h3>
          </div>
        </div>
        
        {transactions.length === 0 ? (
          <div className="no-transactions">
            <p>No transactions yet. Add funds to get started!</p>
          </div>
        ) : (
          <div className="transactions-list">
            {transactions.slice(0, 10).map(transaction => (
              <div key={transaction.id} className="transaction-item">
                <div className="transaction-info">
                  <div className="transaction-type">
                    {transaction.type === 'payment' ? 'Funds Added' : 
                     transaction.type === 'processing' ? 'Video Processing' : 
                     transaction.type}
                  </div>
                  <div className="transaction-date">
                    {formatDate(transaction.createdAt)}
                  </div>
                </div>
                <div className={`transaction-amount ${transaction.type}`}>
                  {transaction.type === 'payment' ? '+' : '-'}
                  {formatCurrency(Math.abs(transaction.amount))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default AccountBalance
