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
 *
 * filter_audio() never blocks: it copies samples into acw_core's lock-free
 * rings and reads the delayed output from them. Everything slow (network, TLS,
 * Opus, reconnects) happens on acw_core's own threads.
 */

#include <obs-module.h>
#include <util/platform.h>
#include <plugin-support.h>

#include "caption-output.h"

#include "acw/reference.hpp"
#include "acw/session.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
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

#define DEFAULT_URL "wss://mc-multi-world.space/audio/api/stream/ws"
#define ECHO_DESKTOP "@desktop"  // the Desktop Audio source (output channel 1); "" = no echo cancellation

#define T_(s) obs_module_text(s)

namespace {

constexpr size_t kMaxBlock = 4096;  // frames handled per inner step of filter_audio()

struct Settings {
	std::string api_key;
	std::string url = DEFAULT_URL;
	bool denoise = true;
	std::string captions = "off";  // off | local | premium
	std::string language = "auto";
	int strength = 100;            // attenuation limit in dB: 12 / 24 / 40 / 100 (same levels as the website and desktop app)
	int latency_ms = 150;          // 80 / 150 / 250
	std::string codec = "opus";    // opus | pcm
	bool send_captions = false;
	std::string echo_source = ECHO_DESKTOP;  // name of the source that plays through the speakers

	bool configured() const { return !api_key.empty() && (denoise || captions != "off"); }

	// Changes that need a new session (the output delay D is fixed per session).
	bool same_session_as(const Settings &o) const
	{
		return api_key == o.api_key && url == o.url && denoise == o.denoise && captions == o.captions &&
		       language == o.language && latency_ms == o.latency_ms && codec == o.codec &&
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

	// "Send captions to stream" can be toggled without restarting the session.
	std::shared_ptr<std::atomic<bool>> forward = std::make_shared<std::atomic<bool>>(false);
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
	s.language = obs_data_get_string(d, S_LANGUAGE);
	if (s.language.empty())
		s.language = "auto";
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

// Called with the lock held. Creates the (not yet started) session for the current settings.
acw::Session *make_session(ac_filter *f)
{
	const Settings &s = f->settings;
	if (!f->enabled || !s.configured())
		return nullptr;

	acw::SessionConfig cfg;
	cfg.url = s.url;
	cfg.api_key = s.api_key;
	cfg.denoise = s.denoise;
	cfg.captions = s.captions == "off" ? "" : s.captions;
	cfg.language = s.language == "auto" ? "" : s.language;
	cfg.atten_lim_db = s.strength;
	cfg.delay_ms = s.latency_ms;
	cfg.codec = s.codec == "pcm" ? acw::Codec::Pcm16 : acw::Codec::Opus;
	cfg.host_rate = static_cast<int>(f->sample_rate);
	if (!s.echo_source.empty())
		cfg.reference = f->reference;
	cfg.client = std::string("obs-audio-cleaner/") + PLUGIN_VERSION;

	f->forward->store(s.send_captions);
	std::shared_ptr<std::atomic<bool>> forward = f->forward;
	return new acw::Session(
		cfg,
		[forward](const acw::CaptionEvent &ev) {
			if (forward->load())
				caption_output_push(ev.text.c_str());
		},
		log_cb);
}

// ---------------------------------------------------------------------------
// echo reference

// Runs on whichever thread delivers the source's audio. OBS audio is planar float at the OBS rate.
void reference_audio_cb(void *param, obs_source_t *, const struct audio_data *audio, bool muted)
{
	auto *t = static_cast<ref_tap *>(param);
	if (muted || audio->frames == 0)
		return;  // a muted source is not playing: no reference
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
	}
	// else: the source is being destroyed, and libobs may still call the callback for a moment.
	// The tap (a few KB) is left behind rather than risk a call into freed memory.
	obs_weak_source_release(f->ref_weak);
	f->ref_weak = nullptr;
	f->tap = nullptr;
}

// Makes the listening match the settings: the chosen source, or none. Called with the lock held;
// also repeated about once a second (filter_tick) because the source may not exist yet when the
// scene is loading, or may be recreated.
void update_reference(ac_filter *f)
{
	const std::string &want = f->settings.echo_source;
	if (!f->enabled || !f->reference || want.empty()) {
		detach_reference(f);
		return;
	}
	if (f->ref_weak && f->ref_name == want && !obs_weak_source_expired(f->ref_weak))
		return;
	detach_reference(f);

	obs_source_t *src = nullptr;
	if (want == ECHO_DESKTOP) {
		src = obs_get_output_source(1);  // Desktop Audio
	} else {
		src = obs_get_source_by_name(want.c_str());
	}
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
	f->retry_s = 0.0f;
	update_reference(f);
}

// Stops and frees a session that is no longer reachable through f->session.
// Waits (bounded) for an in-flight filter_audio() call that may still use it.
void retire_session(ac_filter *f, acw::Session *old)
{
	if (!old)
		return;
	for (int i = 0; i < 200 && f->in_audio.load(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	if (f->in_audio.load()) {
		// The audio thread is still inside a call: do not free under its feet.
		f->graveyard.push_back(old);
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
	obs_data_set_default_bool(d, S_SEND_CAPTIONS, false);
	obs_data_set_default_string(d, S_ECHO_SOURCE, ECHO_DESKTOP);
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
			s->stop(0, 100);
			delete s;
		}
		f->graveyard.clear();
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
	f->forward->store(next.send_captions);
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
		if (!s->started())
			s->start();  // lazy start: only once the enabled filter really receives audio

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

std::string status_text(ac_filter *f)
{
	std::lock_guard<std::mutex> lk(f->mu);
	if (!f->enabled)
		return T_("AudioCleaner.Status.Disabled");
	if (f->settings.api_key.empty())
		return T_("AudioCleaner.Status.NoKey");
	if (!f->settings.configured())
		return T_("AudioCleaner.Status.NothingEnabled");
	acw::Session *s = f->session.load();
	if (!s)
		return "Not running";
	std::string line = s->status_line();
	if (!f->settings.echo_source.empty() && !f->ref_weak)
		line += " | echo cancellation: waiting for the speaker source";
	return line;
}

bool refresh_status_clicked(obs_properties_t *props, obs_property_t *, void *data)
{
	auto *f = static_cast<ac_filter *>(data);
	if (f)
		if (obs_property_t *st = obs_properties_get(props, S_STATUS))
			obs_property_set_description(st, status_text(f).c_str());
	return true;  // redraw the dialog with the new text
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

	obs_property_t *key = obs_properties_add_text(p, S_API_KEY, T_("AudioCleaner.ApiKey"), OBS_TEXT_PASSWORD);
	obs_property_set_long_description(key, T_("AudioCleaner.ApiKey.Long"));

	obs_property_t *url = obs_properties_add_text(p, S_SERVER_URL, T_("AudioCleaner.ServerUrl"), OBS_TEXT_DEFAULT);
	obs_property_set_long_description(url, T_("AudioCleaner.ServerUrl.Long"));

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
		if (f) {
			std::lock_guard<std::mutex> lk(f->mu);
			ctx.current = f->settings.echo_source;
		}
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

	obs_property_t *lang = obs_properties_add_list(p, S_LANGUAGE, T_("AudioCleaner.Language"), OBS_COMBO_TYPE_EDITABLE,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(lang, T_("AudioCleaner.Language.Long"));
	static const char *langs[] = {"auto", "en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl",
				      "cs",   "ru", "uk", "tr", "ar", "hi", "ja", "ko", "zh"};
	for (const char *l : langs)
		obs_property_list_add_string(lang, l, l);

	obs_property_t *send = obs_properties_add_bool(p, S_SEND_CAPTIONS, T_("AudioCleaner.SendCaptions"));
	obs_property_set_long_description(send, T_("AudioCleaner.SendCaptions.Long"));

	obs_property_t *lat = obs_properties_add_list(p, S_LATENCY, T_("AudioCleaner.Latency"), OBS_COMBO_TYPE_LIST,
						      OBS_COMBO_FORMAT_INT);
	obs_property_set_long_description(lat, T_("AudioCleaner.Latency.Long"));
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Low"), 80);
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Balanced"), 150);
	obs_property_list_add_int(lat, T_("AudioCleaner.Latency.Safe"), 250);

	obs_property_t *codec = obs_properties_add_list(p, S_CODEC, T_("AudioCleaner.Codec"), OBS_COMBO_TYPE_LIST,
							OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(codec, T_("AudioCleaner.Codec.Opus"), "opus");
	obs_property_list_add_string(codec, T_("AudioCleaner.Codec.Pcm"), "pcm");

	// Read-only status: for OBS_TEXT_INFO the dialog shows the description when the
	// stored value is empty, so the text is never written to the scene file.
	const std::string st = f ? status_text(f) : std::string();
	obs_property_t *status = obs_properties_add_text(p, S_STATUS, st.c_str(), OBS_TEXT_INFO);
	obs_property_text_set_info_word_wrap(status, true);
	obs_properties_add_button2(p, S_REFRESH, T_("AudioCleaner.RefreshStatus"), refresh_status_clicked, f);
	return p;
}

}  // namespace

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
