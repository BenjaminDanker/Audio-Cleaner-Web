// The editor's icons (the prototype's own drawings).

const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 2.2, strokeLinecap: 'round', strokeLinejoin: 'round' };
const Ic = ({ d, children }) => <svg className="ico" viewBox="0 0 24 24" {...stroke} aria-hidden="true">{d ? <path d={d} /> : children}</svg>;

export const IcPlay = () => <svg className="ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z" /></svg>;
export const IcPause = () => <svg className="ico" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4h4v16H6zM14 4h4v16h-4z" /></svg>;
export const IcSpark = () => <svg className="sp" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2l2.2 7.8L22 12l-7.8 2.2L12 22l-2.2-7.8L2 12l7.8-2.2z" /></svg>;
export const IcCheck = () => <Ic d="M5 12.5l4.5 4.5L19 7" />;
export const IcUndo = () => <Ic><path d="M9 14L4 9l5-5" /><path d="M4 9h10a6 6 0 010 12h-3" /></Ic>;
export const IcRedo = () => <Ic><path d="M15 14l5-5-5-5" /><path d="M20 9H10a6 6 0 000 12h3" /></Ic>;
export const IcSplit = () => <Ic><path d="M12 3v18" /><path d="M5 8l-2 4 2 4M19 8l2 4-2 4" /></Ic>;
export const IcAll = () => <Ic d="M4 7V4h3M17 4h3v3M20 17v3h-3M7 20H4v-3" />;
export const IcSpeaker = () => <Ic><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" /><path d="M15.5 9a4 4 0 010 6M18.5 6.5a7.5 7.5 0 010 11" /></Ic>;
export const IcSpeakerOff = () => <Ic><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" /><path d="M16 9.5l5 5M21 9.5l-5 5" /></Ic>;
export const IcMore = () => <Ic><rect x="3" y="5" width="12" height="12" rx="2" strokeDasharray="3 2.2" /><path d="M19 4v6M16 7h6" /></Ic>;
export const IcChevron = () => <Ic d="M6 9l6 6 6-6" />;
export const IcFile = () => <Ic><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z" /><path d="M14 3v5h5" /></Ic>;
export const IcZoomIn = () => <Ic><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4.2-4.2M11 8.2v5.6M8.2 11h5.6" /></Ic>;
export const IcZoomOut = () => <Ic><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4.2-4.2M8.2 11h5.6" /></Ic>;
