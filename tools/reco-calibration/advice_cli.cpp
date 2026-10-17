// Build:  g++ -std=c++17 -O2 -I obs-plugin/core/include tools/reco-calibration/advice_cli.cpp obs-plugin/core/src/input_advice.cpp -o $RECO_ROOT/reco/advice_cli
// Runs acw::InputAnalyser on mono/stereo 48 kHz WAV files (PCM16 / float32), whole file as "microphone" (LIMIT_S: only the
// first seconds). Prints: file, speech heard (1/0), clipped fraction. The noise figure is not the plugin's: the live engine measures it.
#include "acw/input_advice.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>
using namespace acw;
static bool read_wav(const char *p, std::vector<float> &out) {
	std::ifstream f(p, std::ios::binary); if (!f) return false;
	std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
	size_t off = 12; uint16_t tag = 0, ch = 1, bits = 16;
	while (off + 8 <= b.size()) {
		uint32_t sz; std::memcpy(&sz, &b[off + 4], 4);
		if (!std::memcmp(&b[off], "fmt ", 4)) { std::memcpy(&tag, &b[off + 8], 2); std::memcpy(&ch, &b[off + 10], 2); std::memcpy(&bits, &b[off + 22], 2); }
		if (!std::memcmp(&b[off], "data", 4)) {
			size_t n = std::min<size_t>(sz, b.size() - off - 8) / (bits / 8) / ch;
			out.resize(n);
			for (size_t i = 0; i < n; ++i) { double s = 0; for (int c = 0; c < ch; ++c) { size_t o = off + 8 + (i * ch + c) * (bits / 8); float v; if (tag == 3) std::memcpy(&v, &b[o], 4); else { int16_t q; std::memcpy(&q, &b[o], 2); v = q / 32768.f; } s += v; } out[i] = static_cast<float>(s / ch); }
			return true;
		}
		off += 8 + sz + (sz & 1);
	}
	return false;
}
int main(int argc, char **argv) {
	for (int a = 1; a < argc; ++a) {
		std::vector<float> x; if (!read_wav(argv[a], x)) { std::fprintf(stderr, "cannot read %s\n", argv[a]); continue; }
		const size_t lim = std::min(x.size(), std::getenv("LIMIT_S") ? static_cast<size_t>(std::atof(std::getenv("LIMIT_S")) * 48000) : x.size());
		InputAnalyser an(static_cast<double>(lim / 480) / 100.0);  // all of it
		for (size_t i = 0; i + 960 <= lim; i += 960) an.push(x.data() + i, 960);
		InputFindings f = an.findings();
		std::printf("%s %d %.6f\n", argv[a], f.has_speech ? 1 : 0, f.clip_fraction);
	}
}
