from processor.src.captions.caption_encoder import write_srt, write_vtt, Segment
import tempfile
import os

def test_write_srt_and_vtt_basic():
    segs = [Segment(0.0, 1.5, "Hello world."), Segment(1.5, 3.0, "Testing 1 2 3.")]
    with tempfile.TemporaryDirectory() as td:
        srt = os.path.join(td, 'out.srt')
        vtt = os.path.join(td, 'out.vtt')
        write_srt(segs, srt)
        write_vtt(segs, vtt)
        assert os.path.exists(srt)
        assert os.path.exists(vtt)
        assert 'Hello world.' in open(srt, 'r', encoding='utf-8').read()
        assert 'WEBVTT' in open(vtt, 'r', encoding='utf-8').read()
