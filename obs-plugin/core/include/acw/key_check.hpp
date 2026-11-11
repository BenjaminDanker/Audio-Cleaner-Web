// SPDX-License-Identifier: GPL-2.0-or-later
// Is this API key good? Asked of the live service without starting a session, so that a wrong key shows when it is
// typed in and not at the moment of going live.
//
// The connection is opened exactly as a session opens it (the service decides about the key right there) and closed
// again before anything is started: nothing runs, nothing is billed. Every question counts toward the hourly limit on
// connections, so each server and key is asked once and the answer is remembered until the plugin unloads; when the
// service could not be reached there is no answer, and the pair can be asked again.
#pragma once

#include <string>

namespace acw {

enum class KeyVerdict {
	Unknown,   // not asked, no answer yet, or the service could not be reached
	Accepted,
	Wrong,     // the service does not know this key (mistyped, deleted)
	Refused,   // the key is not for live audio, or the account is not active
};

// Asks in a background thread and returns at once. Nothing happens for an empty key, an address a session would not
// use either (see server_url_allowed), a pair that is being asked right now, or one that has its answer (unless
// `again`: then the answer is forgotten and asked for anew). `client` is the User-Agent, `ca_file` as in SessionConfig.
void ask_key(const std::string &url, const std::string &api_key, const std::string &client, bool again = false,
	     const std::string &ca_file = std::string());

// The answer for this pair so far.
KeyVerdict key_verdict(const std::string &url, const std::string &api_key);

// True while the question for this pair is on its way.
bool key_check_running(const std::string &url, const std::string &api_key);

// Ends the questions that are on their way and waits up to `ms` for their threads; none can be asked afterwards. Call
// when the plugin unloads, before its code can go away. True when no thread is left.
bool stop_key_checks(int ms);

}  // namespace acw
