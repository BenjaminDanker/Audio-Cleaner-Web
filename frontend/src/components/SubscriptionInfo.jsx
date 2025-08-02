import React from 'react'
import { useSubscription } from './SubscriptionContext'
import { CreditCard, Calendar, Users, Zap } from 'lucide-react'
import './SubscriptionInfo.css'

const SubscriptionInfo = () => {
  const { subscription, refreshSubscription, loading } = useSubscription()

  const handleRefresh = () => {
    refreshSubscription()
  }

  if (!subscription && loading) {
    return (
      <div className="subscription-loading">
        <p>Loading subscription information...</p>
      </div>
    )
  }

  if (!subscription) {
    return (
      <div className="subscription-loading">
        <p>No subscription information available.</p>
        <button onClick={handleRefresh} className="btn btn-primary">
          Load Subscription
        </button>
      </div>
    )
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString()
  }

  const getUsagePercentage = () => {
    if (!subscription.limits?.monthly_videos || subscription.limits.monthly_videos === 0) {
      return 0
    }
    return Math.round((subscription.usage?.videos_this_month || 0) / subscription.limits.monthly_videos * 100)
  }

  const getRemainingVideos = () => {
    const used = subscription.usage?.videos_this_month || 0
    const limit = subscription.limits?.monthly_videos || 0
    return Math.max(0, limit - used)
  }

  return (
    <div className="subscription-info">
      <div className="subscription-header">
        <h2>Subscription Details</h2>
        <button onClick={handleRefresh} className="btn btn-secondary">
          Refresh
        </button>
      </div>

      <div className="subscription-cards">
        <div className="subscription-card plan-card">
          <div className="card-header">
            <CreditCard size={24} />
            <h3>Current Plan</h3>
          </div>
          <div className="plan-info">
            <div className="plan-name">{subscription.plan_name || 'Free Plan'}</div>
            <div className="plan-price">${subscription.monthly_cost || 0}/month</div>
            <div className="plan-status">
              Status: <span className={`status ${subscription.status}`}>
                {subscription.status || 'active'}
              </span>
            </div>
          </div>
        </div>

        <div className="subscription-card usage-card">
          <div className="card-header">
            <Zap size={24} />
            <h3>Monthly Usage</h3>
          </div>
          <div className="usage-info">
            <div className="usage-stats">
              <div className="stat">
                <span className="stat-value">{subscription.usage?.videos_this_month || 0}</span>
                <span className="stat-label">Videos Processed</span>
              </div>
              <div className="stat">
                <span className="stat-value">{getRemainingVideos()}</span>
                <span className="stat-label">Remaining</span>
              </div>
            </div>
            
            <div className="usage-progress">
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${getUsagePercentage()}%` }}
                ></div>
              </div>
              <div className="progress-text">
                {getUsagePercentage()}% of {subscription.limits?.monthly_videos || 0} videos used
              </div>
            </div>
          </div>
        </div>

        <div className="subscription-card billing-card">
          <div className="card-header">
            <Calendar size={24} />
            <h3>Billing Information</h3>
          </div>
          <div className="billing-info">
            <div className="billing-item">
              <span className="label">Next Billing Date:</span>
              <span className="value">
                {subscription.next_billing_date ? 
                  formatDate(subscription.next_billing_date) : 
                  'N/A'
                }
              </span>
            </div>
            <div className="billing-item">
              <span className="label">Renewal:</span>
              <span className="value">
                {subscription.auto_renewal ? 'Automatic' : 'Manual'}
              </span>
            </div>
            <div className="billing-item">
              <span className="label">Member Since:</span>
              <span className="value">
                {subscription.created_date ? 
                  formatDate(subscription.created_date) : 
                  'N/A'
                }
              </span>
            </div>
          </div>
        </div>

        <div className="subscription-card features-card">
          <div className="card-header">
            <Users size={24} />
            <h3>Plan Features</h3>
          </div>
          <div className="features-list">
            <div className="feature">
              ✓ Up to {subscription.limits?.monthly_videos || 0} videos per month
            </div>
            <div className="feature">
              ✓ AI-powered noise reduction
            </div>
            <div className="feature">
              ✓ Multiple video formats supported
            </div>
            <div className="feature">
              ✓ High-quality audio output
            </div>
            {subscription.limits?.priority_processing && (
              <div className="feature">
                ✓ Priority processing
              </div>
            )}
            {subscription.limits?.bulk_processing && (
              <div className="feature">
                ✓ Bulk processing
              </div>
            )}
          </div>
        </div>
      </div>

      {getRemainingVideos() === 0 && (
        <div className="usage-warning">
          <h3>Monthly Limit Reached</h3>
          <p>You've used all your videos for this month. Upgrade your plan or wait until next month to process more videos.</p>
          <button className="btn btn-primary">Upgrade Plan</button>
        </div>
      )}
    </div>
  )
}

export default SubscriptionInfo
