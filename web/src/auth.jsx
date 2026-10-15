import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, fetchMe, onUnauthorized } from './api.js';

const AuthContext = createContext(null);

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [state, setState] = useState({ ready: false, known: false, signedIn: false, user: null, balance: null, autoReload: null, freeSecondsLeft: 0 });
  const stateRef = useRef(state);
  stateRef.current = state;

  const apply = useCallback((me) => {
    if (me?.signedIn) {
      setState({ ready: true, known: true, signedIn: true, user: me.user, balance: me.balance, autoReload: me.autoReload, freeSecondsLeft: Math.max(0, Number(me.freeTranscribeSecondsLeft) || 0) });
    } else {
      setState({ ready: true, known: true, signedIn: false, user: null, balance: null, autoReload: null, freeSecondsLeft: 0 });
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const me = await fetchMe();
      apply(me);
      return me;
    } catch {
      // Network or server trouble: keep what we have (`known` stays false until the server has answered once, so the
      // header does not offer "Sign in" to someone who may well be signed in), but unblock the UI.
      setState((s) => ({ ...s, ready: true }));
      return null;
    }
  }, [apply]);

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
