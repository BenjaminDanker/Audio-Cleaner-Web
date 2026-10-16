// SPDX-License-Identifier: GPL-2.0-or-later
// The languages of live captions and translations: codes, English names, what the closed-caption format can show.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace acw {

struct LanguageInfo {
	const char *code;  // 2-letter code, the same as the website's language list
	const char *name;  // English name (used when the plugin has no translation of its own)
};

// The languages a live session can translate into (up to kMaxTranslations at once), in the order they are offered.
const std::vector<LanguageInfo> &translation_languages();
constexpr size_t kMaxTranslations = 3;

// English name of a code ("es" -> "Spanish"); the code itself when unknown.
std::string language_name(const std::string &code);

// The closed-caption format of a stream (CEA-608) can only show Latin letters, and only the accents of the western
// European languages: captions in other scripts can be made, but not added to the stream.
bool language_fits_stream(const std::string &code);

// "en-US", "DE", " fr_CA " -> "en", "de", "fr"; anything that is not a 2-3 letter code -> "auto".
std::string normalise_language(const std::string &in);

// The language the speaker uses, as far as the plugin knows it: Standard captions are English only, extra accurate
// ones have the chosen language ("" = detected automatically, not known in advance).
std::string spoken_language(const std::string &captions, const std::string &language);

// Cleans a list of translation targets: lower case, only known codes, no duplicates, none that equals the spoken
// language (the service refuses it), at most kMaxTranslations. Empty entries ("None") are dropped.
std::vector<std::string> clean_translation_targets(const std::vector<std::string> &wanted, const std::string &spoken);

}  // namespace acw
