import React, { useEffect, useState } from 'react'
import axios from 'axios'

const ApiKeys = () => {
  const [creating, setCreating] = useState(false)
  const [lastSecret, setLastSecret] = useState(null)

  const createKey = async () => {
    setCreating(true)
    try {
      const res = await axios.post('/api/api-keys')
      setLastSecret(res.data.apiKey)
    } catch (e) {
      console.error(e)
      alert('Failed to create key')
    } finally {
      setCreating(false)
    }
  }

  const copySecret = async () => {
    if (!lastSecret) return
    try {
      await navigator.clipboard.writeText(lastSecret)
      // Optionally clear after a short delay to reduce exposure
      setTimeout(() => setLastSecret(null), 5000)
    } catch (e) {
      console.error('Copy failed', e)
      alert('Failed to copy to clipboard')
    }
  }

  return (
    <div>
      <h2>API Keys</h2>
      <p>Use this key with the OBS companion. You’ll see the secret only right after creating/rotating.</p>
      <button disabled={creating} onClick={createKey}>{creating ? 'Working…' : 'Create / Rotate API Key'}</button>
      {lastSecret && (
        <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
          <code style={{ padding: '6px 8px', background: '#111', color: '#0f0', borderRadius: 4, userSelect: 'all' }}>
            {lastSecret}
          </code>
          <button onClick={copySecret}>Copy</button>
        </div>
      )}
    </div>
  )
}

export default ApiKeys
