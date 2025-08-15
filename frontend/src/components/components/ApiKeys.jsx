import React, { useEffect, useState } from 'react'
import axios from 'axios'

const ApiKeys = () => {
  const [keys, setKeys] = useState([])
  const [creating, setCreating] = useState(false)
  const [lastSecret, setLastSecret] = useState(null)

  const load = async () => {
    try {
      const res = await axios.get('/api/api-keys')
      setKeys(res.data.keys || [])
    } catch (e) {
      console.error(e)
      alert('Failed to load API keys')
    }
  }

  useEffect(() => { load() }, [])

  const createKey = async () => {
    setCreating(true)
    try {
      const res = await axios.post('/api/api-keys', { name: 'obs' })
      setLastSecret(res.data.apiKey)
      await load()
    } catch (e) {
      console.error(e)
      alert('Failed to create key')
    } finally {
      setCreating(false)
    }
  }

  const revoke = async (id) => {
    if (!window.confirm('Revoke this key?')) return
    try {
      await axios.delete(`/api/api-keys/${id}`)
      await load()
    } catch (e) {
      console.error(e)
      alert('Failed to revoke key')
    }
  }

  return (
    <div>
      <h2>API Keys</h2>
      <p>Use these keys with the OBS companion. Only newly created keys show their secret once.</p>
      <button disabled={creating} onClick={createKey}>{creating ? 'Creating…' : 'Create API Key'}</button>
      {lastSecret && (
        <div style={{ marginTop: 10 }}>
          <strong>Copy now:</strong>
          <code style={{ marginLeft: 8 }}>{lastSecret}</code>
        </div>
      )}
      <ul style={{ marginTop: 16 }}>
        {keys.map(k => (
          <li key={k.id} style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <span>{k.name} — {k.isActive ? 'active' : 'revoked'}</span>
            {k.isActive && <button onClick={() => revoke(k.id)}>Revoke</button>}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default ApiKeys
