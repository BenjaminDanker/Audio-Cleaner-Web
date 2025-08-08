import React, { useEffect } from 'react'
import axios from 'axios'
import { useAccount } from './AccountContext'
import { CreditCard, DollarSign, Upload, History } from 'lucide-react'
import './AccountBalance.css'

const AccountBalance = () => {
  const { account, loading, refreshAccount, loadAccount } = useAccount()
  const [transactions, setTransactions] = React.useState([])

  // Trigger load if user navigated here before prefetch completed
  useEffect(() => {
    if (!account && !loading) {
      loadAccount()
    }
  }, [account, loading, loadAccount])

  // We still need transactions (not stored in context) so fetch them when account becomes available.
  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        if (!account) return
        const res = await fetch('/api/get-account-data')
        if (res.ok) {
          const data = await res.json()
          setTransactions(data.transactions || [])
        }
      } catch (e) {
        console.error('Failed to fetch transactions:', e)
      }
    }
    fetchTransactions()
  }, [account])

  const handleAddFunds = async () => {
    // Just redirect to Stripe - let them handle everything
    try {
      const response = await axios.post('/api/create-payment-session', {
        // Let Stripe handle amount selection
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

  if (loading || !account) {
    return (
      <div className="account-balance loading-state">
        <h2>Account Balance</h2>
        <p>Loading your account data...</p>
      </div>
    )
  }

  return (
    <div className="account-balance">
      <div className="account-header">
        <h2>Account Balance</h2>
  <button onClick={refreshAccount} className="btn btn-secondary">
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
            <p className="add-funds-description">
              Add money to your account using Stripe's secure checkout
            </p>
            <button 
              onClick={handleAddFunds}
              className="btn btn-primary"
            >
              <Upload size={16} />
              Add Funds via Stripe
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
              <span className="pricing-value">$0.50 per GB</span>
            </div>
            <div className="pricing-item">
              <span className="pricing-label">Minimum charge:</span>
              <span className="pricing-value">$0.05</span>
            </div>
            <div className="pricing-note">
              Cost is calculated based on file size. Perfect for small test files!
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
                     transaction.type === 'refund' ? 'Refund' :
                     transaction.type}
                  </div>
                  <div className="transaction-date">
                    {formatDate(transaction.createdAt)}
                  </div>
                </div>
                <div className={`transaction-amount ${transaction.type}`}>
                  {transaction.type === 'payment' || transaction.type === 'refund' ? '+' : '-'}
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
