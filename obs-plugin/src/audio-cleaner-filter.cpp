/*
Audio Cleaner (cloud) - OBS Studio plugin
Copyright (C) 2026 Audio Cleaner contributors

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

/*
 * "Audio Cleaner (cloud)": an audio filter that streams the source to the
 * Audio Cleaner service (docs/streaming-protocol.md), removes noise and/or
 * produces live captions, and returns the cleaned audio delayed by a fixed
 * amount D. The output timestamp is back-dated by D so OBS keeps audio and
 * video in sync.
 *
 * Echo cancellation: the filter can also listen to another source (usually Desktop
 * Audio: what the speakers play) through an audio capture callback and send that
 * to the service as the echo reference, so the speakers' sound is taken out of the
 * microphone. The reference crosses to the audio thread in a lock-free buffer.
 * libobs hands that callback the audio before the source's mute and volume are
 * applied (the `muted` flag only says what OBS itself will do), so a source that is
 * muted in the OBS mixer but still plays through the speakers keeps being sent.
 *
 * filter_audio() never blocks: it copies samples into acw_core's lock-free
 * rings and reads the delayed output from them. Everything slow (network, TLS,
 * Opus, reconnects) happens on acw_core's own threads.
 */

#include <obs-module.h>
#include <util/platform.h>
#include <plugin-support.h>

#include "caption-output.h"
#include "update-notice.h"

#include "acw/reference.hpp"
#include "acw/session.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define S_API_KEY "api_key"
#define S_SERVER_URL "server_url"
#define S_DENOISE "denoise"
#define S_CAPTIONS "captions"
#define S_LANGUAGE "language"
#define S_STRENGTH "strength"
#define S_LATENCY "latency_ms"
#define S_CODEC "codec"
#define S_SEND_CAPTIONS "send_captions"
#define S_ECHO_SOURCE "echo_source"
#define S_STATUS "status"
#define S_REFRESH "refresh_status"
#define S_RETRY "retry_now"
#define S_ADVANCED "advanced"
#define S_DETAILS "details"
#define S_CHECK_UPDATES "check_updates"  // shows the plugin-wide setting (update-notice.cpp); the filter's copy is unused
#define S_UPDATE_INFO "update_info"
#define S_UPDATE_DOWNLOAD "update_download"

#define DEFAULT_URL "wss://mc-multi-world.space/audio/api/stream/ws"
#define ECHO_DESKTOP "@desktop"  // the Desktop Audio source (output channel 1); "" = no echo cancellation

#define T_(s) obs_module_text(s)

namespace {

constexpr size_t kMaxBlock = 4096;  // frames handled per inner step of filter_audio()

// Languages offered for captions: ISO 639-1 code and the locale key of its name.
struct LangEntry {
	const char *code;
	const char *key;
};
const LangEntry kLanguages[] = {
	{"auto", "AudioCleaner.Language.auto"}, {"en", "AudioCleaner.Language.en"}, {"de", "AudioCleaner.Language.de"},
	{"fr", "AudioCleaner.Language.fr"},     {"es", "AudioCleaner.Language.es"}, {"it", "AudioCleaner.Language.it"},
	{"pt", "AudioCleaner.Language.pt"},     {"nl", "AudioCleaner.Language.nl"}, {"sv", "AudioCleaner.Language.sv"},
	{"da", "AudioCleaner.Language.da"},     {"no", "AudioCleaner.Language.no"}, {"fi", "AudioCleaner.Language.fi"},
	{"pl", "AudioCleaner.Language.pl"},     {"cs", "AudioCleaner.Language.cs"}, {"tr", "AudioCleaner.Language.tr"},
	{"ru", "AudioCleaner.Language.ru"},     {"uk", "AudioCleaner.Language.uk"}, {"ar", "AudioCleaner.Language.ar"},
	{"hi", "AudioCleaner.Language.hi"},     {"ja", "AudioCleaner.Language.ja"}, {"ko", "AudioCleaner.Language.ko"},
	{"zh", "AudioCleaner.Language.zh"},
};

// The closed-caption format of a stream (CEA-608) can only show Latin letters: captions in these languages
// can be made, but not added to the stream.
bool language_fits_stream(const std::string &code)
{
	static const char *no[] = {"ru", "uk", "ar", "hi", "ja", "ko", "zh", "he", "el", "th", "bg", "sr", "fa", "bn", "ta", "te"};
	for (const char *n : no)
		if (code == n)
			return false;
	return true;
}

// "en-US", "DE", " fr_CA " -> "en", "de", "fr"; anything that is not a 2-3 letter code -> "auto".
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

struct Settings {
	std::string api_key;
	std::string url = DEFAULT_URL;
	bool denoise = true;
	std::string captions = "off";  // off | local (standard, English) | premium (extra accurate, any language)
	std::string language = "auto";
	int strength = 100;            // attenuation limit in dB: 12 / 24 / 40 / 100 (same levels as the website and desktop app)
	int latency_ms = 150;          // 80 / 150 / 250
	std::string codec = "opus";    // opus | pcm
	bool send_captions = true;
	std::string echo_source = ECHO_DESKTOP;  // name of the source that plays through the speakers

	// Standard captions are English only; the language setting belongs to the extra accurate ones.
	std::string caption_language() const { return captions == "premium" ? language : std::string("auto"); }

	// Captions only have a use when they go to the stream, and only then are they asked for (and billed).
	bool captions_wanted() const { return captions != "off"; }
	bool captions_fit_stream() const { return language_fits_stream(caption_language()); }
	std::string effective_captions() const
	{
		return captions_wanted() && send_captions && captions_fit_stream() ? captions : std::string();
	}

	bool configured() const { return !api_key.empty() && (denoise || !effective_captions().empty()); }

	// Changes that need a new session (the output delay D is fixed per session).
	bool same_session_as(const Settings &o) const
	{
		const std::string cap = effective_captions();
		return api_key == o.api_key && url == o.url && denoise == o.denoise && cap == o.effective_captions() &&
		       (cap.empty() || caption_language() == o.caption_language()) && latency_ms == o.latency_ms &&
		       codec == o.codec &&
		       echo_source.empty() == o.echo_source.empty();  // which source it is does not matter to the session
	}
};

// What the capture callback of the echo source needs. Freed only once libobs can no longer call it.
struct ref_tap {
	std::shared_ptr<acw::ReferenceBuffer> buffer;
	std::vector<float> mono;  // scratch for the downmix (callback thread only)
};

struct ac_filter {
	obs_source_t *context = nullptr;
	uint32_t sample_rate = 48000;
	size_t channels = 2;

	std::mutex mu;  // serialises create/update/enable/destroy (never taken by the audio thread)
	Settings settings;
	bool enabled = true;

	// The audio thread only ever reads `session`.
	std::atomic<acw::Session *> session{nullptr};
	std::atomic<bool> in_audio{false};
	std::vector<acw::Session *> graveyard;  // sessions that could not be freed safely yet

	std::vector<float> mono;  // preallocated scratch for the downmix

	// Echo reference: the buffer lives as long as the filter, sessions and the tap share it.
	std::shared_ptr<acw::ReferenceBuffer> reference;
	obs_weak_source_t *ref_weak = nullptr;  // the source we listen to (weak: never keeps it alive)
	ref_tap *tap = nullptr;
	std::string ref_name;                   // echo_source the tap was set up for
	float retry_s = 0.0f;

	// Taps of sources that were being destroyed when we let go of them: freed a little later, when libobs
	// can no longer call them.
	struct RetiredTap {
		ref_tap *tap;
		float age_s;
	};
	std::vector<RetiredTap> retired_taps;
};

Settings read_settings(obs_data_t *d)
{
	Settings s;
	s.api_key = obs_data_get_string(d, S_API_KEY);
	s.url = obs_data_get_string(d, S_SERVER_URL);
	if (s.url.empty())
		s.url = DEFAULT_URL;
	s.denoise = obs_data_get_bool(d, S_DENOISE);
	s.captions = obs_data_get_string(d, S_CAPTIONS);
	if (s.captions != "local" && s.captions != "premium")
		s.captions = "off";
	s.language = normalise_language(obs_data_get_string(d, S_LANGUAGE));
	s.strength = static_cast<int>(obs_data_get_int(d, S_STRENGTH));
	s.latency_ms = static_cast<int>(obs_data_get_int(d, S_LATENCY));
	if (s.latency_ms != 80 && s.latency_ms != 250)
		s.latency_ms = 150;
	s.codec = obs_data_get_string(d, S_CODEC) == std::string("pcm") ? "pcm" : "opus";
	s.send_captions = obs_data_get_bool(d, S_SEND_CAPTIONS);
	s.echo_source = obs_data_get_string(d, S_ECHO_SOURCE);
	return s;
}

void log_cb(acw::LogLevel lv, const std::string &msg)
{
	int level = LOG_INFO;
	switch (lv) {
	case acw::LogLevel::Debug:
		level = LOG_DEBUG;
		break;
	case acw::LogLevel::Info:
		level = LOG_INFO;
		break;
	case acw::LogLevel::Warning:
		level = LOG_WARNING;
		break;
	case acw::LogLevel::Error:
		level = LOG_ERROR;
		break;
	}
	obs_log(level, "%s", msg.c_str());
}

// Called with the lock held. Creates the session for the current settings and starts its network thread
// (it only connects once audio flows, so this costs nothing while the filter is idle).
acw::Session *make_session(ac_filter *f)
{
	const Settings &s = f->settings;
	if (!f->enabled || !s.configured())
		return nullptr;

	acw::SessionConfig cfg;
	cfg.url = s.url;
	cfg.api_key = s.api_key;
	cfg.denoise = s.denoise;
	cfg.captions = s.effective_captions();
	cfg.language = cfg.captions.empty() || s.caption_language() == "auto" ? "" : s.caption_language();
	cfg.atten_lim_db = s.strength;
	cfg.delay_ms = s.latency_ms;
	cfg.codec = s.codec == "pcm" ? acw::Codec::Pcm16 : acw::Codec::Opus;
	cfg.host_rate = static_cast<int>(f->sample_rate);
	if (!s.echo_source.empty())
		cfg.reference = f->reference;
	cfg.client = std::string("obs-audio-cleaner/") + PLUGIN_VERSION;

	auto *session = new acw::Session(
		cfg, [](const acw::CaptionEvent &ev) { caption_output_push(ev.text.c_str()); }, log_cb);
	session->start();
	return session;
}

// ---------------------------------------------------------------------------
// echo reference

// Runs on whichever thread delivers the source's audio. OBS audio is planar float at the OBS rate.
void reference_audio_cb(void *param, obs_source_t *, const struct audio_data *audio, bool /*muted*/)
{
	auto *t = static_cast<ref_tap *>(param);
	// `muted` is ignored on purpose: it only says that OBS will not mix this source into the stream. The
	// computer still plays it through the speakers (a common setup: Desktop Audio muted in the mixer so the
	// game stays off the stream), and the microphone still hears it, so it is still the echo reference.
	if (audio->frames == 0)
		return;
	size_t nch = 0;
	const float *planes[MAX_AV_PLANES];
	for (size_t c = 0; c < MAX_AV_PLANES && audio->data[c]; ++c)
		planes[nch++] = reinterpret_cast<const float *>(audio->data[c]);
	if (nch == 0)
		return;
	const float inv = 1.0f / static_cast<float>(nch);
	for (size_t done = 0; done < audio->frames;) {
		const size_t n = std::min<size_t>(audio->frames - done, kMaxBlock);
		float *mono = t->mono.data();
		for (size_t i = 0; i < n; ++i) {
			float acc = planes[0][done + i];
			for (size_t c = 1; c < nch; ++c)
				acc += planes[c][done + i];
			mono[i] = acc * inv;
		}
		t->buffer->push(mono, n);
		done += n;
	}
}

// Stops listening to the echo source. Called with the lock held.
void detach_reference(ac_filter *f)
{
	if (!f->ref_weak)
		return;
	if (obs_source_t *src = obs_weak_source_get_source(f->ref_weak)) {
		// Returns once a running callback has finished: the tap can go.
		obs_source_remove_audio_capture_callback(src, reference_audio_cb, f->tap);
		obs_source_release(src);
		delete f->tap;
	} else if (f->tap) {
		// The source is being destroyed, and libobs may still call the callback for a moment: free the tap
		// a few seconds from now instead of risking a call into freed memory.
		f->retired_taps.push_back({f->tap, 0.0f});
	}
	obs_weak_source_release(f->ref_weak);
	f->ref_weak = nullptr;
	f->tap = nullptr;
}

// Frees retired taps that are old enough. `all`: at shutdown, also the ones that are not (they are then
// left behind rather than freed under a possibly running callback).
void reap_retired_taps(ac_filter *f, float elapsed_s, bool all)
{
	for (auto it = f->retired_taps.begin(); it != f->retired_taps.end();) {
		it->age_s += elapsed_s;
		if (it->age_s >= 10.0f) {
			delete it->tap;
			it = f->retired_taps.erase(it);
		} else if (all) {
			it = f->retired_taps.erase(it);  // leaked on purpose
		} else {
			++it;
		}
	}
}

// The source the settings point at right now (a new reference), or null.
obs_source_t *find_echo_source(const std::string &want)
{
	return want == ECHO_DESKTOP ? obs_get_output_source(1)  // Desktop Audio
				    : obs_get_source_by_name(want.c_str());
}

// Makes the listening match the settings: the chosen source, or none. Called with the lock held;
// also repeated about once a second (filter_tick) because the source may not exist yet when the
// scene is loading, or may be recreated, renamed or replaced (another device chosen for Desktop Audio).
void update_reference(ac_filter *f)
{
	const std::string &want = f->settings.echo_source;
	if (!f->enabled || !f->reference || want.empty()) {
		detach_reference(f);
		return;
	}
	obs_source_t *src = find_echo_source(want);
	if (f->ref_weak && f->ref_name == want && src && obs_weak_source_references_source(f->ref_weak, src)) {
		obs_source_release(src);
		return;  // still the same source
	}
	detach_reference(f);
	if (!src)
		return;
	if (src == obs_filter_get_parent(f->context)) {  // a source cannot be its own speakers
		obs_source_release(src);
		return;
	}
	f->tap = new ref_tap{f->reference, std::vector<float>(kMaxBlock, 0.0f)};
	f->ref_weak = obs_source_get_weak_source(src);
	f->ref_name = want;
	obs_source_add_audio_capture_callback(src, reference_audio_cb, f->tap);
	obs_source_release(src);
}

// Frees sessions that had to wait for the audio thread. Called with the lock held.
void reap_graveyard(ac_filter *f)
{
	if (f->graveyard.empty() || f->in_audio.load(std::memory_order_seq_cst))
		return;
	// Not inside filter_audio() now, and every later call sees the newer session (seq_cst): safe to free.
	for (acw::Session *old : f->graveyard) {
		old->stop(0, 300);
		delete old;
	}
	f->graveyard.clear();
}

// Graphics thread, every frame: retry the listening about once a second.
void filter_tick(void *data, float seconds)
{
	auto *f = static_cast<ac_filter *>(data);
	f->retry_s += seconds;
	if (f->retry_s < 1.0f)
		return;
	std::unique_lock<std::mutex> lk(f->mu, std::try_to_lock);
	if (!lk.owns_lock())
		return;
	const float elapsed = f->retry_s;
	f->retry_s = 0.0f;
	update_reference(f);
	reap_retired_taps(f, elapsed, false);
	reap_graveyard(f);
}

// Ends a session that is no longer reachable through f->session. Called with the lock held.
// The connection (and its billing) ends at once; the object is freed as soon as no filter_audio() call
// that may still use it is running, immediately in practice.
void retire_session(ac_filter *f, acw::Session *old)
{
	if (!old)
		return;
	old->request_stop();
	if (f->in_audio.load(std::memory_order_seq_cst)) {
		f->graveyard.push_back(old);  // freed by the next tick
		return;
	}
	old->stop(0, 300);
	delete old;
}

// Called with the lock held.
void rebuild_session(ac_filter *f)
{
	acw::Session *fresh = make_session(f);
	acw::Session *old = f->session.exchange(fresh);
	retire_session(f, old);
	reap_graveyard(f);
	update_reference(f);
}

void on_enable_signal(void *data, calldata_t *cd)
{
	auto *f = static_cast<ac_filter *>(data);
	const bool enabled = calldata_bool(cd, "enabled");
	std::lock_guard<std::mutex> lk(f->mu);
	if (f->enabled == enabled)
		return;
	f->enabled = enabled;
	rebuild_session(f);  // disabled: stop the session (and billing); enabled: a new one starts lazily
}

const char *filter_name(void *)
{
	return T_("AudioCleaner.FilterName");
}

void filter_defaults(obs_data_t *d)
{
	obs_data_set_default_string(d, S_API_KEY, "");
	obs_data_set_default_string(d, S_SERVER_URL, DEFAULT_URL);
	obs_data_set_default_bool(d, S_DENOISE, true);
	obs_data_set_default_string(d, S_CAPTIONS, "off");
	obs_data_set_default_string(d, S_LANGUAGE, "auto");
	obs_data_set_default_int(d, S_STRENGTH, 100);
	obs_data_set_default_int(d, S_LATENCY, 150);
	obs_data_set_default_string(d, S_CODEC, "opus");
	obs_data_set_default_bool(d, S_SEND_CAPTIONS, true);
	obs_data_set_default_string(d, S_ECHO_SOURCE, ECHO_DESKTOP);
	obs_data_set_default_bool(d, S_CHECK_UPDATES, true);
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *f = new ac_filter();
	f->context = source;
	f->mono.assign(kMaxBlock, 0.0f);

	audio_t *audio = obs_get_audio();
	if (audio) {
		f->sample_rate = audio_output_get_sample_rate(audio);
		f->channels = audio_output_get_channels(audio);
	}
	if (f->sample_rate >= 8000 && f->sample_rate <= 48000)  // the rates the sessions support
		f->reference = std::make_shared<acw::ReferenceBuffer>(static_cast<int>(f->sample_rate));

	{
		std::lock_guard<std::mutex> lk(f->mu);
		f->settings = read_settings(settings);
		// libobs sets source->enabled = true only *after* create() returns, so
		// obs_source_enabled() is still false here. Filters start enabled; a saved
		// disabled state arrives afterwards through the "enable" signal.
		f->enabled = true;
		rebuild_session(f);
	}
	signal_handler_connect(obs_source_get_signal_handler(source), "enable", on_enable_signal, f);
	obs_add_tick_callback(filter_tick, f);
	return f;
}

void filter_destroy(void *data)
{
	auto *f = static_cast<ac_filter *>(data);
	signal_handler_disconnect(obs_source_get_signal_handler(f->context), "enable", on_enable_signal, f);
	obs_remove_tick_callback(filter_tick, f);  // waits for a tick in progress
	{
		std::lock_guard<std::mutex> lk(f->mu);
		f->enabled = false;
		rebuild_session(f);  // stops the session
		for (acw::Session *s : f->graveyard) {
			s->stop(0, 300);
			delete s;
		}
		f->graveyard.clear();
		detach_reference(f);
		reap_retired_taps(f, 0.0f, true);
	}
	delete f;
}

void filter_update(void *data, obs_data_t *settings)
{
	auto *f = static_cast<ac_filter *>(data);
	const Settings next = read_settings(settings);
	std::lock_guard<std::mutex> lk(f->mu);
	const Settings prev = f->settings;
	f->settings = next;
	if (!prev.same_session_as(next)) {
		rebuild_session(f);
		return;
	}
	if (prev.strength != next.strength) {
		// Strength can change while streaming: no new session, no change of the delay.
		if (acw::Session *s = f->session.load())
			s->set_atten_lim_db(next.strength);
	}
	update_reference(f);  // a different echo source
}

// ---------------------------------------------------------------------------
// audio thread

obs_audio_data *filter_audio(void *data, obs_audio_data *audio)
{
	auto *f = static_cast<ac_filter *>(data);

	f->in_audio.store(true, std::memory_order_seq_cst);
	acw::Session *s = f->session.load(std::memory_order_seq_cst);
	if (s && audio->frames > 0) {
		float *planes[MAX_AV_PLANES];
		size_t nch = 0;
		for (size_t c = 0; c < MAX_AV_PLANES && audio->data[c]; ++c)
			planes[nch++] = reinterpret_cast<float *>(audio->data[c]);

		if (nch > 0) {
			float *mono = f->mono.data();
			const float inv = 1.0f / static_cast<float>(nch);
			size_t done = 0;
			while (done < audio->frames) {
				const size_t n = std::min<size_t>(audio->frames - done, kMaxBlock);
				for (size_t i = 0; i < n; ++i) {
					float acc = planes[0][done + i];
					for (size_t c = 1; c < nch; ++c)
						acc += planes[c][done + i];
					mono[i] = acc * inv;
				}
				s->process(mono, n);
				for (size_t c = 0; c < nch; ++c)
					std::copy(mono, mono + n, planes[c] + done);
				done += n;
			}
			// The output is the input delayed by D: back-date it so OBS keeps A/V in sync.
			const double delay = s->delay_seconds();
			if (delay > 0.0) {
				const uint64_t d = static_cast<uint64_t>(delay * 1e9);
				audio->timestamp = audio->timestamp > d ? audio->timestamp - d : 0;
			}
		}
	}
	f->in_audio.store(false, std::memory_order_seq_cst);
	return audio;
}

// ---------------------------------------------------------------------------
// properties

// The plain-words status shown in the dialog. Called without the lock.
std::string status_text(ac_filter *f)
{
	std::lock_guard<std::mutex> lk(f->mu);
	const Settings &st = f->settings;
	if (!f->enabled)
		return T_("AudioCleaner.Status.Disabled");
	if (st.api_key.empty())
		return T_("AudioCleaner.Status.NoKey");
	if (!st.configured()) {
		if (!st.denoise && st.captions_wanted() && !st.captions_fit_stream())
			return T_("AudioCleaner.Status.CaptionsLanguage");
		if (!st.denoise && st.captions_wanted() && !st.send_captions)
			return T_("AudioCleaner.Status.CaptionsNotSent");
		return T_("AudioCleaner.Status.NothingEnabled");
	}
	acw::Session *s = f->session.load();
	if (!s)
		return T_("AudioCleaner.Status.NotRunning");
	std::string line = s->status_line();
	if (st.captions_wanted() && !st.captions_fit_stream())
		line += std::string(" ") + T_("AudioCleaner.Status.CaptionsLanguage");
	else if (st.captions_wanted() && !st.send_captions)
		line += std::string(" ") + T_("AudioCleaner.Status.CaptionsNotSent");
	if (!st.echo_source.empty() && !f->ref_weak)
		line += std::string(" ") + T_("AudioCleaner.Status.EchoWaiting");
	return line;
}

std::string details_text(ac_filter *f)
{
	std::lock_guard<std::mutex> lk(f->mu);
	acw::Session *s = f->session.load();
	return s ? s->status_details() : std::string();
}

// Puts the current status into the dialog.
void refresh_texts(ac_filter *f, obs_properties_t *props)
{
	if (obs_property_t *st = obs_properties_get(props, S_STATUS))
		obs_property_set_description(st, status_text(f).c_str());
	if (obs_property_t *adv = obs_properties_get(props, S_ADVANCED))
		if (obs_properties_t *inner = obs_property_group_content(adv))
			if (obs_property_t *d = obs_properties_get(inner, S_DETAILS))
				obs_property_set_description(d, details_text(f).c_str());
}

bool refresh_status_clicked(obs_properties_t *props, obs_property_t *, void *data)
{
	if (auto *f = static_cast<ac_filter *>(data))
		refresh_texts(f, props);
	return true;  // redraw the dialog with the new text
}

// After the user fixed what the server complained about (balance, terms, ...).
bool retry_clicked(obs_properties_t *props, obs_property_t *, void *data)
{
	auto *f = static_cast<ac_filter *>(data);
	if (!f)
		return false;
	{
		std::lock_guard<std::mutex> lk(f->mu);  // the session cannot be freed meanwhile
		if (acw::Session *s = f->session.load())
			s->retry_now();
	}
	refresh_texts(f, props);
	return true;
}

// Which controls make sense follows the captions choice: extra accurate captions have a language to pick,
// standard captions are English only. Sending to the stream only matters when there are captions.
bool captions_modified(obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
	const std::string mode = obs_data_get_string(settings, S_CAPTIONS);
	if (obs_property_t *lang = obs_properties_get(props, S_LANGUAGE))
		obs_property_set_enabled(lang, mode == "premium");
	if (obs_property_t *send = obs_properties_get(props, S_SEND_CAPTIONS))
		obs_property_set_enabled(send, mode == "local" || mode == "premium");
	return true;
}

// "Download the update": opens the release page in the browser. Nothing is downloaded by the plugin.
bool update_download_clicked(obs_properties_t *, obs_property_t *, void *)
{
	update_notice_open_download_page();
	return false;
}

// The checkbox belongs to the whole plugin, not to this filter: it is saved in the plugin's own config file.
bool check_updates_modified(obs_properties_t *, obs_property_t *, obs_data_t *settings)
{
	const bool on = obs_data_get_bool(settings, S_CHECK_UPDATES);
	if (on != update_notice_enabled())
		update_notice_set_enabled(on);
	return false;
}

std::string replace_all(std::string text, const std::string &from, const std::string &to)
{
	for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
		text.replace(at, from.size(), to);
	return text;
}

struct echo_list_ctx {
	obs_property_t *list;
	obs_source_t *self;     // the filtered source: it cannot be its own speakers
	obs_source_t *desktop;  // offered as "Desktop Audio" already
	std::string current;    // the saved choice, kept in the list even when that source is missing
	bool found = false;
};

bool add_audio_source(void *param, obs_source_t *src)
{
	auto *c = static_cast<echo_list_ctx *>(param);
	if (obs_source_get_type(src) != OBS_SOURCE_TYPE_INPUT || !(obs_source_get_output_flags(src) & OBS_SOURCE_AUDIO))
		return true;
	if (src == c->self || src == c->desktop)
		return true;
	const char *name = obs_source_get_name(src);
	obs_property_list_add_string(c->list, name, name);
	c->found = c->found || c->current == name;
	return true;
}

obs_properties_t *filter_properties(void *data)
{
	auto *f = static_cast<ac_filter *>(data);
	obs_properties_t *p = obs_properties_create();
	Settings cur;  // what is saved now, to set up the enabled state of the controls
	if (f) {
		std::lock_guard<std::mutex> lk(f->mu);
		cur = f->settings;
	}

	// A newer version of the plugin was found (once per OBS start, see update-notice.cpp): say so first.
	char latest[64];
	if (update_notice_available(latest, sizeof(latest))) {
		const std::string msg = replace_all(replace_all(T_("AudioCleaner.Update.Available"), "%1", latest), "%2", PLUGIN_VERSION);
		obs_property_t *info = obs_properties_add_text(p, S_UPDATE_INFO, msg.c_str(), OBS_TEXT_INFO);
		obs_property_text_set_info_type(info, OBS_TEXT_INFO_WARNING);
		obs_property_text_set_info_word_wrap(info, true);
		obs_properties_add_button2(p, S_UPDATE_DOWNLOAD, T_("AudioCleaner.Update.Download"), update_download_clicked, nullptr);
	}

	obs_property_t *key = obs_properties_add_text(p, S_API_KEY, T_("AudioCleaner.ApiKey"), OBS_TEXT_PASSWORD);
	obs_property_set_long_description(key, T_("AudioCleaner.ApiKey.Long"));

	obs_properties_add_bool(p, S_DENOISE, T_("AudioCleaner.Denoise"));

	obs_property_t *strength = obs_properties_add_list(p, S_STRENGTH, T_("AudioCleaner.Strength"), OBS_COMBO_TYPE_LIST,
							   OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(strength, T_("AudioCleaner.Strength.Gentle"), 12);
	obs_property_list_add_int(strength, T_("AudioCleaner.Strength.Balanced"), 24);
	obs_property_list_add_int(strength, T_("AudioCleaner.Strength.Strong"), 40);
	obs_property_list_add_int(strength, T_("AudioCleaner.Strength.Maximum"), 100);
	obs_property_set_long_description(strength, T_("AudioCleaner.Strength.Long"));

	obs_property_t *echo = obs_properties_add_list(p, S_ECHO_SOURCE, T_("AudioCleaner.Echo"), OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(echo, T_("AudioCleaner.Echo.Long"));
	obs_property_list_add_string(echo, T_("AudioCleaner.Echo.None"), "");
	obs_property_list_add_string(echo, T_("AudioCleaner.Echo.Desktop"), ECHO_DESKTOP);
	{
		echo_list_ctx ctx{echo, f ? obs_filter_get_parent(f->context) : nullptr, obs_get_output_source(1), {}};
		ctx.current = cur.echo_source;
		obs_enum_sources(add_audio_source, &ctx);
		if (!ctx.found && !ctx.current.empty() && ctx.current != ECHO_DESKTOP)
			obs_property_list_add_string(echo, ctx.current.c_str(), ctx.current.c_str());
		obs_source_release(ctx.desktop);  // null-safe
	}

	obs_property_t *cap = obs_properties_add_list(p, S_CAPTIONS, T_("AudioCleaner.Captions"), OBS_COMBO_TYPE_LIST,
						      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(cap, T_("AudioCleaner.Captions.Off"), "off");
	obs_property_list_add_string(cap, T_("AudioCleaner.Captions.Local"), "local");
	obs_property_list_add_string(cap, T_("AudioCleaner.Captions.Premium"), "premium");
	obs_property_set_long_description(cap, T_("AudioCleaner.Captions.Long"));
	obs_property_set_modified_callback(cap, captions_modified);

	obs_property_t *lang = obs_properties_add_list(p, S_LANGUAGE, T_("AudioCleaner.Language"), OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(lang, T_("AudioCleaner.Language.Long"));
	bool lang_known = false;
	for (const LangEntry &l : kLanguages) {
		obs_property_list_add_string(lang, T_(l.key), l.code);
		lang_known = lang_known || cur.language == l.code;
	}
	if (!lang_known)  // a code from an older setting: keep it selectable
		obs_property_list_add_string(lang, cur.language.c_str(), cur.language.c_str());
	obs_property_set_enabled(lang, cur.captions == "premium");

	obs_property_t *send = obs_properties_add_bool(p, S_SEND_CAPTIONS, T_("AudioCleaner.SendCaptions"));
	obs_property_set_long_description(send, T_("AudioCleaner.SendCaptions.Long"));
	obs_property_set_enabled(send, cur.captions_wanted());

	obs_property_t *lat = obs_properties_add_list(p, S_LATENCY, T_("AudioCleaner.Latency"), OBS_COMBO_TYPE_LIST,
						      OBS_COMBO_FORMAT_INT);
	obs_property_set_long_description(lat, T_("AudioCleaner.Latency.Long"));
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Low"), 80);
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Balanced"), 150);
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Safe"), 250);

	// Read-only status: for OBS_TEXT_INFO the dialog shows the description when the
	// stored value is empty, so the text is never written to the scene file.
	const std::string st = f ? status_text(f) : std::string();
	obs_property_t *status = obs_properties_add_text(p, S_STATUS, st.c_str(), OBS_TEXT_INFO);
	obs_property_text_set_info_word_wrap(status, true);
	obs_properties_add_button2(p, S_REFRESH, T_("AudioCleaner.RefreshStatus"), refresh_status_clicked, f);
	obs_properties_add_button2(p, S_RETRY, T_("AudioCleaner.RetryNow"), retry_clicked, f);

	// Things most people never touch.
	obs_properties_t *adv = obs_properties_create();
	obs_property_t *url = obs_properties_add_text(adv, S_SERVER_URL, T_("AudioCleaner.ServerUrl"), OBS_TEXT_DEFAULT);
	obs_property_set_long_description(url, T_("AudioCleaner.ServerUrl.Long"));
	obs_property_t *codec = obs_properties_add_list(adv, S_CODEC, T_("AudioCleaner.Codec"), OBS_COMBO_TYPE_LIST,
							OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(codec, T_("AudioCleaner.Codec.Opus"), "opus");
	obs_property_list_add_string(codec, T_("AudioCleaner.Codec.Pcm"), "pcm");
	const std::string det = f ? details_text(f) : std::string();
	obs_property_t *details = obs_properties_add_text(adv, S_DETAILS, det.c_str(), OBS_TEXT_INFO);
	obs_property_text_set_info_word_wrap(details, true);
	obs_properties_add_group(p, S_ADVANCED, T_("AudioCleaner.Advanced"), OBS_GROUP_NORMAL, adv);

	// Plugin-wide setting: show its current value in this filter's dialog (the filter's own copy may be stale).
	if (f && f->context) {
		if (obs_data_t *own = obs_source_get_settings(f->context)) {
			obs_data_set_bool(own, S_CHECK_UPDATES, update_notice_enabled());
			obs_data_release(own);
		}
	}
	obs_property_t *upd = obs_properties_add_bool(p, S_CHECK_UPDATES, T_("AudioCleaner.Update.Check"));
	obs_property_set_long_description(upd, T_("AudioCleaner.Update.Check.Long"));
	obs_property_set_modified_callback(upd, check_updates_modified);
	return p;
}

}  // namespace

// Called when the plugin unloads: lets network threads that had to be detached finish before the code goes.
extern "C" void audio_cleaner_shutdown(void)
{
	acw::wait_for_background_threads(1500);
}

extern "C" void audio_cleaner_register_filter(void)
{
	static obs_source_info info = {};
	info.id = "audio_cleaner_filter";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_AUDIO;
	info.get_name = filter_name;
	info.create = filter_create;
	info.destroy = filter_destroy;
	info.update = filter_update;
	info.get_defaults = filter_defaults;
	info.get_properties = filter_properties;
	info.filter_audio = filter_audio;
	obs_register_source(&info);
}
