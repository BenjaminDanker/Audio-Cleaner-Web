import { Component } from 'react';

/**
 * Catches anything that throws while a page is drawing (or a page file that could not be loaded) and shows a friendly
 * message with a Reload button instead of a blank screen. Uses plain links only, so it works outside the router too.
 * `resetKey` (for example the current path) clears the error when it changes, so moving to another page works again.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidUpdate(prev) {
    if (this.state.failed && prev.resetKey !== this.props.resetKey) this.setState({ failed: false });
  }

  componentDidCatch(error) {
    // Nothing is sent anywhere; the console is for whoever is looking.
    console.error(error); // eslint-disable-line no-console
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="container page">
        <div className="card login-card stack center" role="alert" data-testid="error-boundary">
          <h1 className="mb0">Something went wrong</h1>
          <p className="muted mb0">This page didn’t load properly. That can happen right after we update the site. Reloading usually fixes it.</p>
          <div className="row row-center">
            <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>Reload</button>
            <a className="btn btn-secondary" href={import.meta.env.BASE_URL}>Go to the start page</a>
          </div>
        </div>
      </div>
    );
  }
}
