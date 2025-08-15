import React, { useEffect, useMemo, useState } from 'react'
// No axios or WS usage here; OBS plugin starts streaming with API key
import { useAccount } from './AccountContext'
import './Streaming.css'

const AVAILABLE_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
]

const Streaming = () => {
  const { account, loading: accountLoading, loadAccount } = useAccount()
  const [selectedLanguages, setSelectedLanguages] = useState(['en'])
  const [status] = useState('idle')

  useEffect(() => {
    if (!account && !accountLoading) {
      loadAccount()
    }
  }, [account, accountLoading, loadAccount])

  const canStart = useMemo(() => {
    return status === 'idle' && (account?.creditsCents ?? 0) > 0
  }, [status, account])

  // Streaming sessions are started by the OBS plugin using an API key.

  return (
    <div className="streaming">
      <div className="streaming-header">
        <h2>Live Streaming via OBS</h2>
        <p>Start/stop streaming from the OBS companion plugin using your API key. This page shows your account balance and translation preferences.</p>
      </div>

      <div className="streaming-controls">
        <div className="control-group">
          <label>Translations</label>
          <div className="language-list">
            {AVAILABLE_LANGUAGES.map((lang) => (
              <label key={lang.code} className="lang-item">
                <input
                  type="checkbox"
                  checked={selectedLanguages.includes(lang.code)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedLanguages((prev) => Array.from(new Set([...prev, lang.code])))
                    } else {
                      setSelectedLanguages((prev) => prev.filter((c) => c !== lang.code))
                    }
                  }}
                />
                <span>{lang.name}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label>Account</label>
          <div className="account-row">
            <span>Credits:</span>
            <strong>{accountLoading ? 'Loading…' : `${Math.round((account?.creditsCents ?? 0) / 100)} USD`}</strong>
          </div>
        </div>
      </div>

      <div className="streaming-actions">
        <div className="session-info">
          <p>
            To stream: install the OBS Companion from the repo's <code>obs-companion</code> folder or build the native plugin.
            Provide an API key via environment variable and start the session from OBS.
          </p>
          <ol>
            <li>Set <code>STREAMING_API_KEY</code> in your OBS environment.</li>
            <li>Run the companion or plugin to create the session and connect.</li>
            <li>Processed audio returns to OBS; captions are sent as deltas.</li>
          </ol>
        </div>
      </div>
    </div>
  )
}

export default Streaming
