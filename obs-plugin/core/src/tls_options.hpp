// SPDX-License-Identifier: GPL-2.0-or-later
// TLS settings shared by everything in acw_core that talks HTTPS / WSS: the certificate chain and the host name
// are always verified. Private to the library (it needs the WebSocket library's headers).
#pragma once

#include <ixwebsocket/IXSocketTLSOptions.h>

#include <fstream>
#include <string>

namespace acw {
namespace detail {

#if defined(ACW_TLS_MBEDTLS) && !defined(_WIN32)
// mbedTLS has no system trust store: use the distribution's CA bundle.
inline std::string find_ca_bundle()
{
	static const char *candidates[] = {
		"/etc/ssl/certs/ca-certificates.crt",  // Debian, Ubuntu
		"/etc/pki/tls/certs/ca-bundle.crt",    // Fedora, RHEL
		"/etc/ssl/ca-bundle.pem",              // openSUSE
		"/etc/ssl/cert.pem",                   // Alpine, macOS
	};
	for (const char *c : candidates)
		if (std::ifstream(c).good())
			return c;
	return {};
}
#endif

// `ca_file` empty = the system's trust store.
inline ix::SocketTLSOptions make_tls_options(const std::string &ca_file)
{
	ix::SocketTLSOptions tls;
	tls.caFile = ca_file.empty() ? "SYSTEM" : ca_file;  // certificate + hostname are always verified
#if defined(ACW_TLS_MBEDTLS) && !defined(_WIN32)
	if (ca_file.empty()) {
		const std::string bundle = find_ca_bundle();
		if (!bundle.empty())
			tls.caFile = bundle;
	}
#endif
	return tls;
}

}  // namespace detail
}  // namespace acw
