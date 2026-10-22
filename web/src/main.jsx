import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router';
import App from './App.jsx';
import { AuthProvider } from './auth.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import { purgeOldDraft } from './studio/draft.js';
import './styles/global.css';

purgeOldDraft(); // work kept across a sign-in is deleted after a day, even if nobody came back for it

const basename = import.meta.env.BASE_URL.replace(/\/$/, '');

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter basename={basename}>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
);
