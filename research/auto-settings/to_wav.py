"""LibriTTS-R parquet (HF mirror) -> per-utterance wav files in the layout gen.py reads:
libri/LibriTTS_R/{dev,test}-clean/<speaker>/<chapter>/<id>.wav (24 kHz PCM, as in the original release)."""
import glob, os, sys
import pyarrow.parquet as pq
B = os.path.expanduser("~/audio-cleaner-train")
n = 0
for pat, sub in [("test.clean-*.parquet", "test-clean"), ("dev.clean-*.parquet", "dev-clean")]:
    for path in sorted(glob.glob(os.path.join(B, "data", pat))):
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["audio", "speaker_id", "chapter_id", "id"]).to_pylist()
            for r in t:
                d = os.path.join(B, "libri", "LibriTTS_R", sub, r["speaker_id"], r["chapter_id"])
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, r["id"] + ".wav"), "wb") as f:
                    f.write(r["audio"]["bytes"])
                n += 1
        print(path, n, flush=True)
print("utterances", n)
