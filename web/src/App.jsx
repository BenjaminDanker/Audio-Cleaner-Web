import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { rememberReturn, takeReturn } from './returnTo.js';
import { useAuth } from './auth.jsx';
import Layout from './components/Layout.jsx';
import NeedsServer from './components/Maintenance.jsx';
import Landing from './pages/Landing.jsx';

const Free = lazy(() => import('./pages/Free.jsx'));
const Login = lazy(() => import('./pages/Login.jsx'));
const LoginVerify = lazy(() => import('./pages/LoginVerify.jsx'));
const Dashboard = lazy(() => import('./pages/Dashboard.jsx'));
const Billing = lazy(() => import('./pages/Billing.jsx'));
const Live = lazy(() => import('./pages/Live.jsx'));
const Account = lazy(() => import('./pages/Account.jsx'));
const Terms = lazy(() => import('./pages/Terms.jsx'));
const Privacy = lazy(() => import('./pages/Privacy.jsx'));
const Licenses = lazy(() => import('./pages/Licenses.jsx'));
const NotFound = lazy(() => import('./pages/NotFound.jsx'));

function Loading() {
  return <div className="loading-screen" role="status"><span className="spinner" aria-hidden="true" /> <span className="sr-only">Loading</span></div>;
}

function RequireAuth({ children }) {
  const { ready, signedIn } = useAuth();
  const loc = useLocation();
  if (!ready) return <Loading />;
  if (!signedIn) {
    rememberReturn(loc.pathname);
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  }
  const back = takeReturn();
  if (back && back !== loc.pathname) return <Navigate to={back} replace />;
  return children;
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="free" element={<NeedsServer><RequireAuth><Free /></RequireAuth></NeedsServer>} />
          <Route path="login" element={<NeedsServer><Login /></NeedsServer>} />
          <Route path="login/verify" element={<LoginVerify />} />
          <Route path="app" element={<NeedsServer><RequireAuth><Dashboard /></RequireAuth></NeedsServer>} />
          <Route path="app/billing" element={<NeedsServer><RequireAuth><Billing /></RequireAuth></NeedsServer>} />
          <Route path="app/live" element={<NeedsServer><RequireAuth><Live /></RequireAuth></NeedsServer>} />
          <Route path="app/account" element={<NeedsServer><RequireAuth><Account /></RequireAuth></NeedsServer>} />
          <Route path="terms" element={<Terms />} />
          <Route path="privacy" element={<Privacy />} />
          <Route path="licenses" element={<Licenses />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
