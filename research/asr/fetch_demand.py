#!/usr/bin/env python
"""Fetch only channel 01 of four DEMAND (CC BY 4.0, Zenodo 1227121) 48 kHz noise zips via
HTTP range requests (the full zips are 250-370 MB each; ch01 is ~20 MB).
Output: ~/audio-cleaner/research/data/raw/demand/<ENV>_ch01.wav  (untrusted data, only read by soundfile).
Run: ~/audio-cleaner/research/venv/bin/python fetch_demand.py
"""
import io, struct, sys, time, urllib.request, zipfile
from pathlib import Path

OUT = Path.home() / "audio-cleaner/research/data/raw/demand"


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = b""
        while len(data) < n:  # robust to short reads / resets
            r = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos + len(data)}-{self.pos + n - 1}"})
            for attempt in range(8):  # Zenodo is flaky (TLS handshake timeouts)
                try:
                    with urllib.request.urlopen(r, timeout=60) as resp:
                        assert resp.status == 206, resp.status
                        data += resp.read()
                    break
                except (OSError, AssertionError):
                    if attempt == 7:
                        raise
                    time.sleep(2 * (attempt + 1))
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def main():
    for env in ["DLIVING", "PCAFETER", "STRAFFIC", "OOFFICE"]:
        dest = OUT / f"{env}_ch01.wav"
        if dest.exists():
            continue
        rf = io.BufferedReader(RangeFile(f"https://zenodo.org/records/1227121/files/{env}_48k.zip"), buffer_size=1 << 20)
        with zipfile.ZipFile(rf) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith("ch01.wav")][0]
            dest.write_bytes(zf.read(name))
        print(env, name, dest.stat().st_size)


if __name__ == "__main__":
    main()
