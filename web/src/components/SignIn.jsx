import { Link, useNavigate } from 'react-router';
import { rememberReturn } from '../returnTo.js';

/** Goes to the sign-in page; after signing in the visitor lands on `path` (a page that opens without an account too). */
export function useGoSignIn() {
  const nav = useNavigate();
  return (path = '/app') => {
    rememberReturn(path);
    nav('/login', { state: { from: path } });
  };
}

/** A link to the sign-in page that brings the visitor back to `to` afterwards. */
export function SignInLink({ to = '/app', className = 'btn btn-secondary btn-sm', children = 'Sign in', ...rest }) {
  return <Link to="/login" state={{ from: to }} className={className} onClick={() => rememberReturn(to)} {...rest}>{children}</Link>;
}
