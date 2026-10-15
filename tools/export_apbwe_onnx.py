#!/usr/bin/env python
"""One-time export of the official AP-BWE 48 kHz generators to ONNX (CPU, dynamic time axis).

Not part of the product: the product only needs onnxruntime + numpy + scipy
(acw_common/bwe.py). This needs torch, in a throw-away venv:

    # requirements (pinned, versions used for the shipped files):
    #   torch==2.5.1+cpu   (pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu)
    #   numpy==2.5.4  onnx==1.23.2  onnxscript==0.7.2  onnxruntime==1.31.0  scipy
    # optional (--reference): matplotlib, soundfile, librosa  (only because the upstream utils.py imports them)

    python -I tools/export_apbwe_onnx.py --weights DIR --out DIR [--reference AP-BWE-repo] [--only 8kto48k]

DIR/weights holds the Google-Drive folder of the AP-BWE README (<name>/g_<name> + config.json).
Third-party code is never imported unless --reference is given (that one is for the fidelity check only).

What is exported: the network only (APNet_BWE_Model.forward minus the final atan2/exp/cos/sin):
    inputs   mag (1,513,T) = log(|STFT|+1e-4)   pha (1,513,T) = angle(STFT)        [float32]
    outputs  mag_wb (1,513,T) log-magnitude, pha_r / pha_i (1,513,T): re/im of the predicted phase
             (angle = atan2(pha_i, pha_r); done in numpy by bwe.py)
The STFT (n_fft 1024, hop 80, win 320 Hann, centre, reflect) and iSTFT are reimplemented in numpy in
bwe.py; this script checks that reimplementation against torch.stft/istft and against the official model.

weights_only=True is used for torch.load (the checkpoints contain only the 'generator' state dict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

NAMES = ("8kto48k", "12kto48k", "16kto48k", "24kto48k")
N_FFT, HOP, WIN = 1024, 80, 320
CH, LAYERS = 512, 8


class Block(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 3)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.gamma = nn.Parameter(scale * torch.ones(dim))

    def forward(self, x):
        r = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        return r + x.transpose(1, 2)


class Net(nn.Module):
    """APNet_BWE_Model (MIT, (c) 2023 Ye-Xin Lu) with the same parameter names, exportable output."""

    def __init__(self):
        super().__init__()
        nb = N_FFT // 2 + 1
        self.conv_pre_mag = nn.Conv1d(nb, CH, 7, 1, padding=3)
        self.norm_pre_mag = nn.LayerNorm(CH, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(nb, CH, 7, 1, padding=3)
        self.norm_pre_pha = nn.LayerNorm(CH, eps=1e-6)
        self.convnext_mag = nn.ModuleList([Block(CH, 1 / LAYERS) for _ in range(LAYERS)])
        self.convnext_pha = nn.ModuleList([Block(CH, 1 / LAYERS) for _ in range(LAYERS)])
        self.norm_post_mag = nn.LayerNorm(CH, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(CH, eps=1e-6)
        self.linear_post_mag = nn.Linear(CH, nb)
        self.linear_post_pha_r = nn.Linear(CH, nb)
        self.linear_post_pha_i = nn.Linear(CH, nb)

    def forward(self, mag_nb, pha_nb):
        xm = self.norm_pre_mag(self.conv_pre_mag(mag_nb).transpose(1, 2)).transpose(1, 2)
        xp = self.norm_pre_pha(self.conv_pre_pha(pha_nb).transpose(1, 2)).transpose(1, 2)
        for bm, bp in zip(self.convnext_mag, self.convnext_pha):
            xm = xm + xp
            xp = xp + xm
            xm = bm(xm)
            xp = bp(xp)
        mag_wb = mag_nb + self.linear_post_mag(self.norm_post_mag(xm.transpose(1, 2))).transpose(1, 2)
        xp = self.norm_post_pha(xp.transpose(1, 2))
        return mag_wb, self.linear_post_pha_r(xp).transpose(1, 2), self.linear_post_pha_i(xp).transpose(1, 2)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_net(ckpt):
    net = Net()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["generator"]
    net.load_state_dict(sd, strict=True)
    return net.eval()


def stft_ref(x):
    w = torch.hann_window(WIN)
    s = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=w, center=True, pad_mode="reflect",
                   normalized=False, return_complex=True)
    return torch.log(s.abs() + 1e-4), torch.angle(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference", type=Path, help="official AP-BWE checkout (fidelity check only)")
    ap.add_argument("--only", nargs="*", default=NAMES)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    torch.set_num_threads(4)
    a.out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    ref_cls = None
    if a.reference:
        sys.path.insert(0, str(a.reference))
        from models.model import APNet_BWE_Model as ref_cls  # noqa: N813
        from env import AttrDict
    rng = np.random.default_rng(0)
    sums = {}
    for name in a.only:
        d = a.weights / name
        ckpt = d / f"g_{name}"
        cfg = json.loads((d / "config.json").read_text())
        assert (cfg["n_fft"], cfg["hop_size"], cfg["win_size"], cfg["hr_sampling_rate"]) == (N_FFT, HOP, WIN, 48000)
        assert (cfg["ConvNeXt_channels"], cfg["ConvNeXt_layers"]) == (CH, LAYERS)
        net = load_net(ckpt)
        onnx_path = a.out / f"{name}.onnx"
        mag = torch.randn(1, N_FFT // 2 + 1, 120)
        pha = torch.randn(1, N_FFT // 2 + 1, 120)
        torch.onnx.export(net, (mag, pha), str(onnx_path), opset_version=a.opset, dynamo=False,
                          input_names=["mag", "pha"], output_names=["mag_wb", "pha_r", "pha_i"],
                          dynamic_axes={k: {2: "T"} for k in ("mag", "pha", "mag_wb", "pha_r", "pha_i")})
        # fidelity: noisy speech-like 3 s at 48 kHz, low-passed by FFT mask at the model's input band
        lr = cfg["lr_sampling_rate"]
        n = 48000 * 3 + 123
        x = rng.standard_normal(n).astype(np.float32) * 0.1
        spec = np.fft.rfft(x)
        spec[int(len(spec) * (lr / 2) / 24000):] = 0
        x = np.fft.irfft(spec, n).astype(np.float32)
        lm, lp = stft_ref(torch.from_numpy(x)[None])
        with torch.no_grad():
            tm, tr, ti = net(lm, lp)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        om, orr, oi = sess.run(None, {"mag": lm.numpy(), "pha": lp.numpy()})
        err = {"mag": float(np.abs(om - tm.numpy()).max()), "re": float(np.abs(orr - tr.numpy()).max()),
               "im": float(np.abs(oi - ti.numpy()).max())}
        # also a different length (dynamic axis) to be sure
        sess.run(None, {"mag": lm.numpy()[:, :, :37], "pha": lp.numpy()[:, :, :37]})
        msg = f"{name}: onnx vs torch max abs err {err}"
        if ref_cls is not None:
            ref = ref_cls(AttrDict(cfg)).eval()
            ref.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["generator"])
            with torch.no_grad():
                rm, rp, rc = ref(lm, lp)
            wav_ref = torch.istft(torch.complex(rc[..., 0], rc[..., 1]), N_FFT, hop_length=HOP, win_length=WIN,
                                  window=torch.hann_window(WIN), center=True)
            msg += f"; my torch net vs official mag err {float((rm - tm).abs().max()):.2e}"
            np.save(a.out / f"_ref_{name}.npy", wav_ref.numpy())
            np.save(a.out / "_ref_in.npy", x)
        print(msg, flush=True)
        sums[onnx_path.name] = (onnx_path.stat().st_size, sha256(onnx_path))
    (a.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, (_, h) in sums.items()))
    print(json.dumps(sums, indent=1))


if __name__ == "__main__":
    main()
