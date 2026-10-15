import { Link } from 'react-router';
import { usePageTitle } from '../hooks.js';

export default function NotFound() {
  usePageTitle('Page not found');
  return (
    <div className="container page error-page">
      <div className="code" aria-hidden="true">404</div>
      <h1>We couldn’t find that page</h1>
      <p className="muted">The link might be old, or a letter slipped in.</p>
      <div className="row row-center">
        <Link to="/" className="btn btn-primary">Back to the start</Link>
        <Link to="/app" className="btn btn-secondary">Try it on a file</Link>
      </div>
    </div>
  );
}
