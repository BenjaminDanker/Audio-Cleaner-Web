"""Build the 10-minute concatenated test file from consecutive VoiceBank-DEMAND *noisy* test utterances
(all 824 sorted by name; 0.3 s of the preceding utterance's noise floor is not added, files are simply joined)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import numpy as np
from common import DATA, SR, VB_NOISY, read_mono, write_wav

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
parts, n = [], 0
for p in sorted(VB_NOISY.glob("*.wav"))[::3]:  # every 3rd utterance: mixes both speakers and noise conditions
    x = read_mono(p)
    parts.append(x); n += len(x)
    if n >= secs * SR:
        break
y = np.concatenate(parts)[: int(secs * SR)]
out = DATA / f"long_{int(secs)}s.wav"
write_wav(out, y)
print(out, len(y) / SR, "s,", len(parts), "utterances")
