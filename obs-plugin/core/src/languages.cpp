// SPDX-License-Identifier: GPL-2.0-or-later
#include "acw/languages.hpp"

#include <algorithm>
#include <cctype>

namespace acw {

const std::vector<LanguageInfo> &translation_languages()
{
	static const std::vector<LanguageInfo> kList = {
		{"en", "English"},    {"es", "Spanish"},  {"fr", "French"},     {"de", "German"},   {"it", "Italian"},
		{"pt", "Portuguese"}, {"nl", "Dutch"},    {"pl", "Polish"},     {"ru", "Russian"},  {"uk", "Ukrainian"},
		{"tr", "Turkish"},    {"ar", "Arabic"},   {"hi", "Hindi"},      {"ja", "Japanese"}, {"ko", "Korean"},
		{"zh", "Chinese"},    {"sv", "Swedish"},  {"da", "Danish"},     {"fi", "Finnish"},  {"no", "Norwegian"},
		{"cs", "Czech"},      {"el", "Greek"},    {"he", "Hebrew"},     {"id", "Indonesian"}, {"vi", "Vietnamese"},
	};
	return kList;
}

std::string language_name(const std::string &code)
{
	for (const LanguageInfo &l : translation_languages())
		if (code == l.code)
			return l.name;
	return code;
}

bool language_fits_stream(const std::string &code)
{
	// Vietnamese is written with Latin letters, but nearly every vowel carries marks the caption format lacks.
	static const char *const kNo[] = {"ru", "uk", "ar", "hi", "ja", "ko", "zh", "he", "el", "th", "bg", "sr", "fa", "bn", "ta", "te", "vi"};
	for (const char *n : kNo)
		if (code == n)
			return false;
	return true;
}

std::string normalise_language(const std::string &in)
{
	std::string out;
	for (char ch : in) {
		const unsigned char c = static_cast<unsigned char>(ch);
		if (std::isspace(c))
			continue;
		if (!std::isalpha(c))
			break;  // '-' or '_' starts the region
		out += static_cast<char>(std::tolower(c));
	}
	if (out.size() < 2 || out.size() > 3)
		return "auto";
	return out;
}

std::string spoken_language(const std::string &captions, const std::string &language)
{
	if (captions == "local")
		return "en";
	if (captions == "premium" && !language.empty() && language != "auto")
		return language;
	return std::string();
}

std::vector<std::string> clean_translation_targets(const std::vector<std::string> &wanted, const std::string &spoken)
{
	std::vector<std::string> out;
	for (const std::string &w : wanted) {
		std::string c;
		for (char ch : w)
			c += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
		bool known = false;
		for (const LanguageInfo &l : translation_languages())
			known = known || c == l.code;
		if (!known || c == spoken || std::find(out.begin(), out.end(), c) != out.end())
			continue;
		if (out.size() >= kMaxTranslations)
			break;
		out.push_back(c);
	}
	return out;
}

}  // namespace acw
