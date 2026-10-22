// A free run made in the visitor's browser has no job on our side: its card has an id of its own, and its page needs no account.
const PREFIX = 'local-';
export const newLocalJobId = () => `${PREFIX}${Date.now().toString(36)}`;
export const isLocalJobId = (id) => typeof id === 'string' && id.startsWith(PREFIX);
