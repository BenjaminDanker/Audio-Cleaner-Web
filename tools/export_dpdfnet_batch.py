#!/usr/bin/env python3
"""Export DPDFNet-8 (48 kHz HR) as a streaming ONNX model with a dynamic batch dimension.

The published `dpdfnet8_48khz_hr.onnx` (Ceva-IP/DPDFNet) is batch-1: `spec (1,1,481,2)`, `state_in (90228,)`, and
its Reshape nodes carry literal batch-1 shapes, so independent segments of one recording cannot share a call. This
script exports the same network, same weights, same maths, with

    spec      (B, 1, 481, 2)  float32   one STFT frame per batch row
    state_in  (B, 90228)      float32   row b is the state of batch row b (same initial values as the batch-1 model)
    -> spec_e (B, 1, 481, 2), state_out (B, 90228)

so row b of a batch-B call equals the batch-1 call on that row. The state of a row is the batch-1 state vector with
the same sections in the same order; only the inside of the FIFO sections (CyclicBuffer) is kept in the layout of the
tensor it feeds (batch, channels, time, bins) instead of time-first, which saves transposes (they start as zeros, so
the initial state is the same as for the batch-1 model). Only the four modules that slice and reshape the state
(CyclicBuffer, GRUCellInternalState, MagNorm48, SpecNorm48), the DPRNN stack (channel-last, fewer transposes) and the
top-level forward are replaced here (monkey patches below). Nothing in the upstream tree is edited.

Needs (not part of the product venv): torch, onnx, onnxsim, onnxruntime, einops, numpy, scipy, and the upstream
code and weights:

    git clone https://github.com/ceva-ip/DPDFNet && git -C DPDFNet checkout 9bd9844a227bb6aa57e55588d8d0e961fcff1c46
    curl -L -o dpdfnet8_48khz_hr.pth \
      https://huggingface.co/Ceva-IP/DPDFNet/resolve/dd6818d00f50c836fed43a6243ebe49116de5964/checkpoints/dpdfnet8_48khz_hr.pth
    python tools/export_dpdfnet_batch.py --upstream DPDFNet --checkpoint dpdfnet8_48khz_hr.pth \
        --output dpdfnet8_48khz_hr_b.onnx --reference ~/audio-cleaner/models/dpdfnet/dpdfnet8_48khz_hr.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def patch(torch, nn):
    """Make the upstream streaming modules batch-aware. State tensors are (n, B): n features, B batch rows."""
    from onnx_model import dpdfnet_48khz_hr as M
    from onnx_model import layers as L

    def cyclic_forward(self, x, state=None, offset=0):
        # Rows keep the buffer in the layout of the input (batch, ..., time = capacity, ...), so no transposes.
        td = self.time_dim
        if x.dim() != len(self.inv_perm) or x.shape[td] != 1:
            raise ValueError("expected one frame along the time dim")
        shape = [self.shape_tf[self.perm.index(d)] for d in range(x.dim())]  # original layout, batch dim = 1
        if state is None:
            buf = x.new_zeros([x.shape[0]] + shape[1:])
        else:
            end = offset + self._state_size
            buf = state[:, offset:end].reshape([-1] + shape[1:])
        buf_next = torch.cat([buf.narrow(td, 1, self.capacity - 1), x], dim=td)
        out = buf_next.narrow(td, 0, self.time_steps)
        if state is None:
            return out
        return out, buf_next.reshape(buf_next.shape[0], self._state_size), offset + self._state_size

    def gru_forward(self, inputs, state=None, offset=0):
        if state is None:
            return self.grucell(inputs, inputs.new_zeros(inputs.shape[0], self.units))
        end = offset + self.state_size()
        h_prev = state[:, offset:end].reshape(-1, self.units)  # row = b * batch_size + f
        y = self.grucell(inputs, h_prev)
        return y, y.reshape(-1, self.state_size()), end

    def mag_forward(self, x, state=None, offset=0):
        if state is None or self.dynamic_var:
            raise NotImplementedError("batched export supports the stateful, static-variance MagNorm48 only")
        end = offset + self.num_feat
        mu_prev = state[:, offset:end].reshape(-1, 1, self.num_feat)
        mu_next = self.alpha * mu_prev + (1 - self.alpha) * x
        x_norm = (x - mu_next) / (self.var0.sqrt() + self.eps)
        return x_norm, mu_next.reshape(-1, self.num_feat), end

    def spec_forward(self, x, state=None, offset=0):
        if state is None:
            raise NotImplementedError("batched export needs the state")
        x_abs = L.get_mag(x)
        end = offset + self.state_size()
        s_prev = state[:, offset:end].reshape(-1, 1, self.num_feat)
        s_next = self.alpha * s_prev + (1 - self.alpha) * x_abs
        denom = (s_next + self.eps).sqrt()
        x_norm = torch.stack([x[..., 0] / denom, x[..., 1] / denom], dim=-1)
        return x_norm, s_next.reshape(-1, self.num_feat), end

    def dprnn_forward(self, inputs, state=None, offset=0):
        # Same arithmetic as DPRNN/DPRNNBlock.forward for one frame (T = 1), but the blocks run channel-last
        # (B, F, C) end to end: the upstream code moves between (B, C, 1, F) and (B, F, C) four times per block
        # and those 4-D transposes were 12 % of the batched model's run time.
        if state is None or inputs.shape[2] != 1:
            raise NotImplementedError("batched export needs the state and one frame")
        assert isinstance(self.input_proj, nn.Identity) and isinstance(self.output_proj, nn.Identity)
        nf, nc = self.blocks[0].num_feat, self.blocks[0].hidden_dim  # static, the exporter must know them
        x = inputs.reshape(-1, nc, nf).transpose(1, 2)  # (B, F, C)
        chunks = []
        for blk in self.blocks:
            xi, _ = blk.intra_gru(x)  # bidirectional over the F bins, batch = B
            x = x + blk.ln_intra(blk.fc_intra(xi)).reshape(-1, nf, nc)
            xt, so, offset = blk.inter_gru(x.reshape(-1, nc), state=state, offset=offset)
            chunks.append(so)
            x = x + blk.ln_inter(blk.fc_inter(xt)).reshape(-1, nf, nc)
        return x.transpose(1, 2).reshape(-1, nc, 1, nf), torch.cat(chunks, dim=1), offset

    def top_forward(self, spec, state):
        # spec (B, 1, F, 2), state (B, S). Same body as upstream DPDFNet48HR.forward (mask_method 'before_df').
        assert self.mask_method == "before_df"
        chunks: List = []
        offset = 0
        spec, feat_erb, feat_spec, st, offset = self._feature_extraction(spec, state=state, offset=offset)
        chunks.append(st)
        feat_spec = feat_spec.permute(0, 3, 1, 2)
        e0, e1, e2, e3, emb, c0, lsnr, st, offset = self.enc(feat_erb, feat_spec, state=state, offset=offset)
        chunks.append(st)
        m, st, offset = self.erb_dec(emb, e3, e2, e1, e0, state=state, offset=offset)
        chunks.append(st)
        df_coefs, st, offset = self.df_dec(emb, c0, state=state, offset=offset)
        chunks.append(st)
        df_coefs = self.df_out_transform(df_coefs)
        spec, st, offset = self.mask(spec, m, state=state, offset=offset)
        chunks.append(st)
        spec_e, st, offset = self.df_op(spec.clone(), df_coefs, state=state, offset=offset)
        chunks.append(st)
        if offset != self.state_size():
            raise RuntimeError(f"State offset mismatch: consumed {offset}, expected {self.state_size()}")
        return spec_e.squeeze(1), torch.cat(chunks, dim=1)

    class TorchProxy:
        """The upstream modules concatenate the state sections with torch.cat(dim=0); with (batch, state) tensors
        that is dim=1. Only 2-D tensors are redirected (the only 2-D concatenations are those state sections)."""
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def cat(tensors, dim=0, **kw):
            if dim == 0 and tensors and tensors[0].dim() == 2:
                dim = 1
            return torch.cat(tensors, dim=dim, **kw)

    from onnx_model import multiframe as MFm
    for mod in (M, L, MFm):
        mod.torch = TorchProxy()
    L.DPRNN.forward = dprnn_forward
    L.CyclicBuffer.forward = cyclic_forward
    L.GRUCellInternalState.forward = gru_forward
    L.MagNorm48.forward = mag_forward
    L.SpecNorm48.forward = spec_forward
    M.DPDFNet48HR.forward = top_forward
    return M, L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="checkout of github.com/ceva-ip/DPDFNet")
    ap.add_argument("--checkpoint", type=Path, required=True, help="checkpoints/dpdfnet8_48khz_hr.pth")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="published batch-1 ONNX file to compare against")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    import torch
    from torch import nn

    sys.path.insert(0, str(args.upstream.resolve()))
    M, L = patch(torch, nn)
    from onnx_model.export_dpdfnet_48khz_hr_to_onnx import (  # noqa: E402
        add_meta_data, build_meta_data, simplify_onnx)

    model = M.DPDFNet48HR(
        conv_kernel_inp=(3, 3), conv_ch=64, enc_gru_dim=256, erb_dec_gru_dim=256, df_dec_gru_dim=256,
        enc_lin_groups=32, lin_groups=16, upsample_conv_type="subpixel", group_linear_type="loop",
        point_wise_type="cnn", separable_first_conv=True, dprnn_num_blocks=8)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(M.correct_state_dict(sd), strict=True)
    L.convert_grouped_linear_to_einsum(model)
    model.eval()

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("wnorm", torch.tensor(float(m.wnorm), dtype=torch.float32))
            self.register_buffer("inv_wnorm", torch.tensor(1.0 / float(m.wnorm), dtype=torch.float32))

        def forward(self, spec, state_in):
            spec_e, state_out = self.m(spec * self.wnorm, state_in)
            return spec_e * self.inv_wnorm, state_out

    wrap = Wrap(model).eval()
    nb = 3
    spec = torch.randn(nb, 1, model.freq_bins, 2)
    st0 = model.initial_state(dtype=torch.float32)
    state = st0[None, :].repeat(nb, 1).contiguous()
    torch.onnx.export(
        wrap, (spec, state), str(args.output), input_names=["spec", "state_in"], output_names=["spec_e", "state_out"],
        opset_version=args.opset, do_constant_folding=True, external_data=False, dynamo=False,
        dynamic_axes={"spec": {0: "batch"}, "state_in": {0: "batch"}, "spec_e": {0: "batch"}, "state_out": {0: "batch"}})
    add_meta_data(args.output, dict(build_meta_data(model, "dpdfnet8_48khz_hr"), batch_layout="state is (batch, state_size)"))
    if not args.no_simplify:
        simplify_onnx(args.output)
        strip_simplifier_metadata(args.output)
    print("wrote", args.output, "state size", model.state_size())
    check(args.output, wrap, model, args.reference)


def strip_simplifier_metadata(path: Path) -> None:
    """onnxsim writes a long change log into the model's metadata; the runtime only needs the model's own keys."""
    import onnx

    m = onnx.load(str(path))
    keep = [(p.key, p.value) for p in m.metadata_props if not p.key.startswith("onnxsim.")]
    del m.metadata_props[:]
    for k, v in keep:
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, str(path))


def check(path: Path, wrap, model, reference: Path | None) -> None:
    """Rows of a batched call must equal the PyTorch model on that row alone, and the published batch-1 file."""
    import onnxruntime as ort
    import torch

    def session(p):
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        return ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])

    new = session(path)
    ref = session(reference) if reference else None
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    init = model.initial_state(dtype=torch.float32)
    for nb in (1, 3, 8):
        x = (rng.standard_normal((60, nb, 1, 481, 2)) * 0.1 * np.linspace(0.2, 3, nb)[None, :, None, None, None]).astype(np.float32)
        state = np.repeat(init.numpy()[None], nb, axis=0)
        ref_states = [init.numpy().copy() for _ in range(nb)]
        worst_t = worst_r = 0.0
        pst = torch.from_numpy(np.repeat(init.numpy()[None], nb, axis=0))
        for t in range(60):
            out, state = new.run(None, {"spec": x[t], "state_in": state})
            with torch.no_grad():  # the same rows through the PyTorch model
                po, pst = wrap(torch.from_numpy(x[t]), pst)
            worst_t = max(worst_t, float(np.abs(po.numpy() - out).max()))
            for b in range(nb if ref is not None else 0):
                o1, ref_states[b] = ref.run(None, {"spec": x[t, b:b + 1], "state_in": ref_states[b]})
                worst_r = max(worst_r, float(np.abs(out[b] - o1[0]).max()))
        print(f"batch {nb}: max abs diff vs PyTorch {worst_t:.2e}" + (f", vs the published batch-1 ONNX {worst_r:.2e}" if ref else ""))
        assert worst_t < 1e-4 and worst_r < 1e-4, "export does not match"


if __name__ == "__main__":
    main()
