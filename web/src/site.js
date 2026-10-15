// Single place for the operator's identity. The site owner fills these in before launch; nothing else in the
// code base should hard-code a contact address, legal entity or jurisdiction.
//
//   name          shown in the header, footer, page titles and legal texts
//   supportEmail  public contact address (footer, Terms, Privacy). Empty = no contact link is shown.
//   legalName     legal entity that operates the service ("Example Ltd")
//   legalAddress  postal address of that entity
//   jurisdiction  governing law / venue for the Terms ("England and Wales", "the State of Delaware, USA")
//
// An empty field is left out: the legal pages then use neutral wording ("the operator") instead.
export const SITE = {
  name: 'Audio Cleaner',
  supportEmail: '',
  legalName: '',
  legalAddress: '',
  jurisdiction: 'the United States',
};

/** "Example Ltd, 1 Main St" / "Example Ltd" / null when nothing is set. */
export function operatorLine() {
  const name = SITE.legalName.trim();
  const addr = SITE.legalAddress.trim();
  if (name && addr) return `${name}, ${addr}`;
  return name || addr || null;
}
