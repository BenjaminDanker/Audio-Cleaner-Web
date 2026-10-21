import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { api, fetchMe, onUnauthorized } from './api.js';

const AuthContext = createContext(null);

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [state, setState] = useState({ ready: false, known: false, signedIn: false, user: null, balance: null, autoReload: null, contact: null, topupMaxCents: null, trialLiveSecondsLeft: 0 });
  const stateRef = useRef(state);
  stateRef.current = state;

  const apply = useCallback((me) => {
    if (me?.signedIn) {
      // `topupMaxCents`: the most one top-up may be right now (the first one is smaller); null = the usual limit. `trialLiveSecondsLeft`: free live trial left.
      setState({ ready: true, known: true, signedIn: true, user: me.user, balance: me.balance, autoReload: me.autoReload, contact: !!me.contactForm, topupMaxCents: me.topupMaxCents ?? null, trialLiveSecondsLeft: me.trialLiveSecondsLeft ?? 0 });
    } else {
      // `contact`: whether the contact form can send mail right now (null = not known yet). Only a yes or no; the address stays on the server.
      setState({ ready: true, known: true, signedIn: false, user: null, balance: null, autoReload: null, contact: me?.contactForm === undefined ? stateRef.current.contact : !!me.contactForm, topupMaxCents: null, trialLiveSecondsLeft: 0 });
    }
  }, []);

  const retry = useRef(0);
  const refresh = useCallback(async () => {
    clearTimeout(retry.current);
    try {
      const me = await fetchMe();
      apply(me);
      return me;
    } catch (e) {
      // Network or server trouble, or too many requests: keep what we have (`known` stays false until the server has answered
      // once, so nobody is sent to the sign-in page for a busy moment), unblock the UI, and while the server has never answered
      // ask again (after the wait it asked for, if it did)
      setState((s) => ({ ...s, ready: true }));
      if (!stateRef.current.known) retry.current = setTimeout(() => refreshRef.current(), Math.min(30, Math.max(2, e?.retryAfter || 3)) * 1000);
      return null;
    }
  }, [apply]);
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  useEffect(() => () => clearTimeout(retry.current), []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    onUnauthorized(() => {
      if (stateRef.current.signedIn) apply({ signedIn: false });
      navigate('/login', { replace: true });
    });
    return () => onUnauthorized(null);
  }, [apply, navigate]);

  const logout = useCallback(async () => {
    try {
      await api('auth/logout', { method: 'POST', redirectOn401: false });
    } catch {
      /* already signed out */
    }
    apply({ signedIn: false });
    navigate('/', { replace: true });
  }, [apply, navigate]);

  const value = useMemo(() => ({ ...state, refresh, logout }), [state, refresh, logout]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
