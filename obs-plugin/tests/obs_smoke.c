/*
 * obs_smoke: headless libobs host that loads the built module and exercises the
 * "Audio Cleaner (cloud)" filter through the real OBS filter chain.
 *
 *   obs_smoke <audio-cleaner.so> <plugin data dir> <ws url> <key file> <input.wav (48k mono s16)> [out.wav]
 *
 * It starts libobs with an audio output, creates an audio input source with the
 * filter attached, feeds the WAV in real time in 10 ms packets (like a capture
 * device), records the mixed output and reports: module API version, property
 * list, the timestamp back-dating (output vs. input timeline), level, and the
 * filter status text. Not a unit test; needs a running server.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <obs.h>
#include <util/base.h>
#include <util/platform.h>
#include <util/threading.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void log_handler(int level, const char *msg, va_list args, void *param)
{
	(void)param;
	if (level > LOG_INFO)
		return;
	char buf[2048];
	vsnprintf(buf, sizeof buf, msg, args);
	/* never echo anything that could be a key */
	printf("  [obs %d] %s\n", level, buf);
}

/* ---- a fake capture device ---- */
static const char *smoke_name(void *d)
{
	(void)d;
	return "smoke input";
}
static void *smoke_create(obs_data_t *s, obs_source_t *src)
{
	(void)s;
	return src;
}
static void smoke_destroy(void *d)
{
	(void)d;
}

/* ---- recording of the filter output ----
 * obs_source_add_audio_capture_callback() delivers every chunk the filter chain produced
 * (after filter_audio) together with its timestamp. Each chunk is written into a timeline
 * at the position its timestamp says, which is exactly where the OBS mixer will put it. */
static uint64_t g_t0;
#define GUARD(rate) ((size_t)(rate))

struct rec {
	float *s;      /* output placed on the capture timeline (index 0 = 1 s before the first input sample) */
	size_t cap, chunks, first_idx, last_idx;
	pthread_mutex_t mu;
};

static void cap_cb(void *param, obs_source_t *source, const struct audio_data *data, bool muted)
{
	(void)source;
	(void)muted;
	struct rec *r = param;
	const int rate = 48000;
	double rel = ((double)data->timestamp - (double)g_t0) / 1e9 * rate;
	long idx = lrint(rel) + (long)GUARD(rate);
	pthread_mutex_lock(&r->mu);
	if (idx >= 0 && (size_t)idx + data->frames <= r->cap) {
		memcpy(r->s + idx, data->data[0], data->frames * sizeof(float));
		if (!r->chunks)
			r->first_idx = (size_t)idx;
		r->last_idx = (size_t)idx + data->frames;
		r->chunks++;
	}
	pthread_mutex_unlock(&r->mu);
}

static float *read_wav(const char *path, size_t *n)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *d = malloc((size_t)sz);
	if (fread(d, 1, (size_t)sz, f) != (size_t)sz) {
		fclose(f);
		return NULL;
	}
	fclose(f);
	/* find "data" chunk (assumes 16-bit mono) */
	size_t pos = 12;
	while (pos + 8 < (size_t)sz) {
		unsigned csz = d[pos + 4] | (d[pos + 5] << 8) | (d[pos + 6] << 16) | ((unsigned)d[pos + 7] << 24);
		if (!memcmp(d + pos, "data", 4)) {
			*n = csz / 2;
			float *x = malloc(*n * sizeof(float));
			for (size_t i = 0; i < *n; i++)
				x[i] = (short)(d[pos + 8 + 2 * i] | (d[pos + 9 + 2 * i] << 8)) / 32768.0f;
			free(d);
			return x;
		}
		pos += 8 + csz + (csz & 1);
	}
	free(d);
	return NULL;
}

static void write_wav(const char *path, const float *x, size_t n, int rate)
{
	FILE *f = fopen(path, "wb");
	if (!f)
		return;
	unsigned bytes = (unsigned)(n * 2), v32;
	unsigned short v16;
	fwrite("RIFF", 1, 4, f);
	v32 = 36 + bytes;
	fwrite(&v32, 4, 1, f);
	fwrite("WAVEfmt ", 1, 8, f);
	v32 = 16;
	fwrite(&v32, 4, 1, f);
	v16 = 1;
	fwrite(&v16, 2, 1, f);
	fwrite(&v16, 2, 1, f);
	v32 = (unsigned)rate;
	fwrite(&v32, 4, 1, f);
	v32 = (unsigned)rate * 2;
	fwrite(&v32, 4, 1, f);
	v16 = 2;
	fwrite(&v16, 2, 1, f);
	v16 = 16;
	fwrite(&v16, 2, 1, f);
	fwrite("data", 1, 4, f);
	fwrite(&bytes, 4, 1, f);
	for (size_t i = 0; i < n; i++) {
		float c = fmaxf(-1.f, fminf(1.f, x[i]));
		short s = (short)lrintf(c * 32767.f);
		fwrite(&s, 2, 1, f);
	}
	fclose(f);
}

static double rms(const float *x, size_t a, size_t b)
{
	double e = 0;
	for (size_t i = a; i < b; i++)
		e += (double)x[i] * x[i];
	return sqrt(e / (double)(b - a));
}

int main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr, "usage: %s <module.so> <data dir> <ws url> <key file> <in.wav> [out.wav]\n", argv[0]);
		return 2;
	}
	const char *module_path = argv[1], *data_dir = argv[2], *url = argv[3], *key_file = argv[4], *wav = argv[5];

	char key[256] = {0};
	FILE *kf = fopen(key_file, "r");
	if (!kf || !fgets(key, sizeof key, kf)) {
		fprintf(stderr, "cannot read key file\n");
		return 2;
	}
	fclose(kf);
	key[strcspn(key, "\r\n")] = 0;

	size_t in_n = 0;
	float *in = read_wav(wav, &in_n);
	if (!in) {
		fprintf(stderr, "cannot read wav\n");
		return 2;
	}
	const int rate = 48000;
	if (in_n > (size_t)rate * 8)
		in_n = (size_t)rate * 8;

	base_set_log_handler(log_handler, NULL);
	printf("libobs %s\n", obs_get_version_string());
	if (!obs_startup("en-US", NULL, NULL)) {
		fprintf(stderr, "obs_startup failed\n");
		return 1;
	}
	struct obs_audio_info ai = {.samples_per_sec = (uint32_t)rate, .speakers = SPEAKERS_STEREO};
	if (!obs_reset_audio(&ai)) {
		fprintf(stderr, "obs_reset_audio failed\n");
		return 1;
	}

	obs_module_t *mod = NULL;
	int rc = obs_open_module(&mod, module_path, data_dir);
	printf("obs_open_module: %s\n", rc == MODULE_SUCCESS ? "ok" : "FAILED");
	if (rc != MODULE_SUCCESS)
		return 1;
	if (!obs_init_module(mod)) {
		fprintf(stderr, "obs_init_module failed\n");
		return 1;
	}
	obs_post_load_modules();
	printf("module description: %s\n", obs_get_module_description(mod));

	/* the filter's registered metadata + property list */
	const char *id = "audio_cleaner_filter";
	printf("filter display name: %s\n", obs_source_get_display_name(id));
	obs_properties_t *props = obs_get_source_properties(id);
	printf("properties:");
	for (obs_property_t *p = obs_properties_first(props); p; obs_property_next(&p))
		printf(" %s", obs_property_name(p));
	printf("\n");
	obs_properties_destroy(props);

	/* input source with the filter */
	struct obs_source_info si = {0};
	si.id = "smoke_input";
	si.type = OBS_SOURCE_TYPE_INPUT;
	si.output_flags = OBS_SOURCE_AUDIO;
	si.get_name = smoke_name;
	si.create = smoke_create;
	si.destroy = smoke_destroy;
	obs_register_source(&si);
	obs_source_t *input = obs_source_create("smoke_input", "mic", NULL, NULL);

	obs_data_t *fs = obs_data_create();
	obs_data_set_string(fs, "api_key", key);
	obs_data_set_string(fs, "server_url", url);
	obs_data_set_bool(fs, "denoise", true);
	obs_data_set_string(fs, "captions", "off");
	obs_data_set_int(fs, "latency_ms", 150);
	obs_source_t *filter = obs_source_create(id, "Audio Cleaner", fs, NULL);
	obs_data_release(fs);
	obs_source_filter_add(input, filter);
	obs_set_output_source(0, input);
	obs_source_set_audio_mixers(input, 0x3F);
	obs_source_set_audio_active(input, true);
	obs_source_set_muted(input, false);
	printf("input source: mixers 0x%x, audio active %d, muted %d, volume %.2f\n", obs_source_get_audio_mixers(input),
	       obs_source_audio_active(input), obs_source_muted(input), obs_source_get_volume(input));

	struct rec r = {0};
	r.cap = (size_t)rate * 30;
	r.s = calloc(r.cap, sizeof(float));
	pthread_mutex_init(&r.mu, NULL);
	obs_source_add_audio_capture_callback(input, cap_cb, &r);

	/* feed in real time */
	const size_t pkt = 480;
	uint64_t t0 = os_gettime_ns();
	g_t0 = t0;
	for (size_t pos = 0; pos + pkt <= in_n; pos += pkt) {
		uint64_t due = t0 + (uint64_t)((double)(pos + pkt) / rate * 1e9);
		uint64_t now = os_gettime_ns();
		if (due > now)
			os_sleepto_ns(due);
		struct obs_source_audio a = {0};
		a.data[0] = (const uint8_t *)(in + pos);
		a.frames = (uint32_t)pkt;
		a.speakers = SPEAKERS_MONO;
		a.format = AUDIO_FORMAT_FLOAT;
		a.samples_per_sec = (uint32_t)rate;
		a.timestamp = t0 + (uint64_t)((double)pos / rate * 1e9); /* capture time of the first sample */
		obs_source_output_audio(input, &a);
	}
	os_sleep_ms(1500);

	/* status as the properties dialog would show it */
	obs_properties_t *p2 = obs_source_properties(filter);
	obs_property_t *st = obs_properties_get(p2, "status");
	printf("status: %s\n", st ? obs_property_description(st) : "(none)");
	obs_properties_destroy(p2);

	obs_source_remove_audio_capture_callback(input, cap_cb, &r);

	/* settings change without a new session (strength), then disable / enable */
	{
		obs_data_t *u = obs_data_create();
		obs_data_set_int(u, "strength", 40);
		obs_source_update(filter, u);
		obs_data_release(u);
		os_sleep_ms(200);
		obs_properties_t *p3 = obs_source_properties(filter);
		printf("after strength change: %s\n", obs_property_description(obs_properties_get(p3, "status")));
		obs_properties_destroy(p3);
		obs_source_set_enabled(filter, false);
		p3 = obs_source_properties(filter);
		printf("after disabling the filter: %s\n", obs_property_description(obs_properties_get(p3, "status")));
		obs_properties_destroy(p3);
		obs_source_set_enabled(filter, true);
		p3 = obs_source_properties(filter);
		printf("after re-enabling (session starts lazily on audio): %s\n",
		       obs_property_description(obs_properties_get(p3, "status")));
		obs_properties_destroy(p3);
	}

	/* Where does the filter put the audio on the timeline? Compare the timestamp-placed output
	 * with the input (indexed by capture time). A filter that delays by D and back-dates its
	 * timestamps lines up with lag ~0; without back-dating the content would sit about +D late. */
	pthread_mutex_lock(&r.mu);
	printf("filter output: %zu chunks, first chunk placed at %.1f ms (relative to the first captured sample)\n", r.chunks,
	       ((double)r.first_idx - (double)GUARD(rate)) * 1000.0 / rate);
	if (r.chunks > 10) {
		const size_t wa = (size_t)rate * 3, wb = (size_t)rate * 5;
		double best = -1e300;
		long best_l = 0;
		for (long l = -(long)rate / 4; l <= (long)rate / 2; l++) {
			double c = 0, e = 1e-9;
			for (size_t i = wa; i < wb && i < in_n; i += 2) {
				long j = (long)(i + GUARD(rate)) + l;
				if (j < 0 || (size_t)j >= r.cap)
					continue;
				c += (double)in[i] * r.s[j];
				e += (double)r.s[j] * r.s[j];
			}
			c /= sqrt(e);
			if (c > best) {
				best = c;
				best_l = l;
			}
		}
		printf("timeline offset of the filter output vs. the captured audio: %+ld samples = %+.2f ms "
		       "(back-dated by D = 150 ms; an un-dated delay would show about +150 ms)\n",
		       best_l, best_l * 1000.0 / rate);
		if (argc > 6)
			write_wav(argv[6], r.s + GUARD(rate), r.last_idx - GUARD(rate), rate);
	}
	pthread_mutex_unlock(&r.mu);

	obs_source_filter_remove(input, filter);
	obs_source_release(filter);
	obs_source_release(input);
	obs_set_output_source(0, NULL);
	obs_shutdown();
	printf("shutdown ok; leaked objects: %ld\n", bnum_allocs());
	return 0;
}
