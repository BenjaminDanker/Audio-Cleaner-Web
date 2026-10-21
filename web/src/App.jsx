import { Suspense, useEffect } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router';
import { lazyRetry as lazy } from './lazy.js';
import { forgetReturn, peekReturn, rememberReturn } from './returnTo.js';
import { useAuth } from './auth.jsx';
import Layout from './components/Layout.jsx';
import NeedsServer from './components/Maintenance.jsx';
import Landing from './pages/Landing.jsx';

const Login = lazy(() => import('./pages/Login.jsx'));
const LoginVerify = lazy(() => import('./pages/LoginVerify.jsx'));
const Cleanup = lazy(() => import('./pages/Cleanup.jsx'));
const Job = lazy(() => import('./pages/Job.jsx'));
const Billing = lazy(() => import('./pages/Billing.jsx'));
const Live = lazy(() => import('./pages/Live.jsx'));
const Account = lazy(() => import('./pages/Account.jsx'));
const Terms = lazy(() => import('./pages/Terms.jsx'));
const Privacy = lazy(() => import('./pages/Privacy.jsx'));
const Licenses = lazy(() => import('./pages/Licenses.jsx'));
const Contact = lazy(() => import('./pages/Contact.jsx'));
const Developers = lazy(() => import('./pages/Developers.jsx'));
const DevelopersLive = lazy(() => import('./pages/DevelopersLive.jsx'));
const NotFound = lazy(() => import('./pages/NotFound.jsx'));

function Loading() {
  return <div className="loading-screen" role="status"><span className="spinner" aria-hidden="true" /> <span className="sr-only">Loading</span></div>;
}

function RequireAuth({ children }) {
  const { ready, known, signedIn } = useAuth();
  const loc = useLocation();
  if (!ready || !known) return <Loading />; // (until the server has answered: a busy moment is not being signed out)
  if (!signedIn) {
    rememberReturn(loc.pathname);
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  }
  return <SignedIn>{children}</SignedIn>;
}

// Signed in: go to the page the visitor wanted before signing in (once), then forget it. Looking at it while drawing is
// harmless; clearing it is done in an effect so a repeated draw (StrictMode) still sees it.
function SignedIn({ children }) {
  const loc = useLocation();
  const back = peekReturn();
  useEffect(() => { forgetReturn(); }, []);
  if (back && back !== loc.pathname) return <Navigate to={back} replace />;
  return children;
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="login" element={<NeedsServer><Login /></NeedsServer>} />
          <Route path="login/verify" element={<LoginVerify />} />
          <Route path="app" element={<NeedsServer><RequireAuth><Cleanup /></RequireAuth></NeedsServer>} />
          <Route path="app/job/:id" element={<NeedsServer><RequireAuth><Job /></RequireAuth></NeedsServer>} />
          <Route path="app/billing" element={<NeedsServer><RequireAuth><Billing /></RequireAuth></NeedsServer>} />
          <Route path="app/live" element={<NeedsServer><RequireAuth><Live /></RequireAuth></NeedsServer>} />
          <Route path="app/account" element={<NeedsServer><RequireAuth><Account /></RequireAuth></NeedsServer>} />
          <Route path="terms" element={<Terms />} />
          <Route path="privacy" element={<Privacy />} />
          <Route path="licenses" element={<Licenses />} />
          <Route path="contact" element={<Contact />} />
          <Route path="developers" element={<Developers />} />
          <Route path="developers/live" element={<DevelopersLive />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
