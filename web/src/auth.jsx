import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { api, fetchMe, onUnauthorized } from './api.js';

const AuthContext = createContext(null);

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [state, setState] = useState({ ready: false, known: false, signedIn: false, user: null, balance: null, autoReload: null, contact: null });
  const stateRef = useRef(state);
  stateRef.current = state;

  const apply = useCallback((me) => {
    if (me?.signedIn) {
      setState({ ready: true, known: true, signedIn: true, user: me.user, balance: me.balance, autoReload: me.autoReload, contact: !!me.contactForm });
    } else {
      // `contact`: whether the contact form can send mail right now (null = not known yet). Only a yes or no; the address stays on the server.
      setState({ ready: true, known: true, signedIn: false, user: null, balance: null, autoReload: null, contact: me?.contactForm === undefined ? stateRef.current.contact : !!me.contactForm });
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
