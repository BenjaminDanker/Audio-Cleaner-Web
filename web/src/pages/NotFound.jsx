import { Link } from 'react-router-dom';
import { usePageTitle } from '../hooks.js';

export default function NotFound() {
  usePageTitle('Page not found');
  return (
    <div className="container page error-page">
      <div className="code" aria-hidden="true">404</div>
      <h1>This page could not be found</h1>
      <p className="muted">The link may be old or mistyped.</p>
      <div className="row row-center">
        <Link to="/" className="btn btn-primary">Back to the home page</Link>
        <Link to="/free" className="btn btn-secondary">Clean a file for free</Link>
      </div>
    </div>
  );
}
