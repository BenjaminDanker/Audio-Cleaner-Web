import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { useAuth } from './auth.jsx';
import Layout from './components/Layout.jsx';
import Landing from './pages/Landing.jsx';

const Free = lazy(() => import('./pages/Free.jsx'));
const Desktop = lazy(() => import('./pages/Desktop.jsx'));
const Login = lazy(() => import('./pages/Login.jsx'));
const LoginVerify = lazy(() => import('./pages/LoginVerify.jsx'));
const Dashboard = lazy(() => import('./pages/Dashboard.jsx'));
const Transcribe = lazy(() => import('./pages/Transcribe.jsx'));
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
  if (!signedIn) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  return children;
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="free" element={<Free />} />
          <Route path="desktop" element={<Desktop />} />
          <Route path="login" element={<Login />} />
          <Route path="login/verify" element={<LoginVerify />} />
          <Route path="app" element={<RequireAuth><Dashboard /></RequireAuth>} />
          <Route path="app/billing" element={<RequireAuth><Billing /></RequireAuth>} />
          <Route path="app/live" element={<RequireAuth><Live /></RequireAuth>} />
          <Route path="app/account" element={<RequireAuth><Account /></RequireAuth>} />
          <Route path="transcribe" element={<Transcribe />} />
          <Route path="terms" element={<Terms />} />
          <Route path="privacy" element={<Privacy />} />
          <Route path="licenses" element={<Licenses />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
