import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, fetchMe, onUnauthorized } from './api.js';
import TosModal from './components/TosModal.jsx';

export const TOS_VERSION = '2026-10-10';

const AuthContext = createContext(null);

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [state, setState] = useState({ ready: false, signedIn: false, user: null, balance: null, autoReload: null });
  const [tosOpen, setTosOpen] = useState(false);
  const tosResolve = useRef(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  const apply = useCallback((me) => {
    if (me?.signedIn) {
      setState({ ready: true, signedIn: true, user: me.user, balance: me.balance, autoReload: me.autoReload });
    } else {
      setState({ ready: true, signedIn: false, user: null, balance: null, autoReload: null });
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const me = await fetchMe();
      apply(me);
      return me;
    } catch {
      // Network or server trouble: keep what we have, but unblock the UI.
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

  /** Resolves true once the current terms are accepted (opens the modal if needed). */
  const ensureTos = useCallback(() => {
    if (stateRef.current.user?.tosAccepted) return Promise.resolve(true);
    return new Promise((resolve) => {
      tosResolve.current = resolve;
      setTosOpen(true);
    });
  }, []);

  const closeTos = useCallback(
    async (accepted) => {
      if (accepted) await refresh();
      setTosOpen(false);
      const r = tosResolve.current;
      tosResolve.current = null;
      if (r) r(accepted);
    },
    [refresh],
  );

  const value = useMemo(() => ({ ...state, refresh, logout, ensureTos }), [state, refresh, logout, ensureTos]);

  return (
    <AuthContext.Provider value={value}>
      {children}
      {tosOpen && <TosModal onClose={closeTos} />}
    </AuthContext.Provider>
  );
}
