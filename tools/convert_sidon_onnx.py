"""Convert the official Sidon TorchScript files (sarulab-speech/sidon-v0.1 @ b3b02d8b, MIT) to ONNX for onnxruntime.

usage: python -I convert_onnx.py MODEL_DIR [OUT_DIR]      (needs torch, onnx, onnxruntime, numpy)
Writes into OUT_DIR (default MODEL_DIR):
  sidon_fe.onnx      feature extractor, (1,T,160) fbank -> (1,T,1024); weights stored as float16, cast to float32 at load
  sidon_dec.onnx     decoder, (1,1024,T) -> (1,1,T*960); same storage
  sidon_fe_q8.onnx   optional: feature extractor with uint8 dynamic-quantised MatMul weights (3x smaller, ~5% feature error)
Compute stays float32 (outputs differ from TorchScript by <1e-3 relative: float16 weight rounding).
Constant nodes that hold weights become initializers first, so onnxruntime can see (and quantise) them.
"""
import sys, warnings
from pathlib import Path

import numpy as np
import onnx, torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

warnings.filterwarnings("ignore")
src = Path(sys.argv[1]); dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
dst.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x)["last_hidden_state"]


def constants_to_initializers(model, min_elems=64):
    keep, inits = [], []
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            t = n.attribute[0].t
            if numpy_helper.to_array(t).size >= min_elems:
                t.name = n.output[0]
                inits.append(t)
                continue
        keep.append(n)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    model.graph.initializer.extend(inits)
    return model


def weights_to_fp16(model, min_elems=64):
    """Store big float32 initializers as float16 plus a Cast back to float32 (folded by onnxruntime at load)."""
    inits, casts = [], []
    for t in model.graph.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32 and a.size >= min_elems:
            inits.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_f16"))
            casts.append(helper.make_node("Cast", [t.name + "_f16"], [t.name], to=TensorProto.FLOAT, name=t.name + "_cast"))
        else:
            inits.append(t)
    nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    del model.graph.node[:]
    model.graph.node.extend(casts + nodes)
    return model


def export(module, example, path, in_name, out_name, in_axis, out_axis):
    tmp = path.with_suffix(".tmp.onnx")
    torch.onnx.export(module, (example,), str(tmp), dynamo=False, input_names=[in_name], output_names=[out_name],
                      dynamic_axes={in_name: {in_axis: "T"}, out_name: {out_axis: "S"}}, opset_version=17)
    m = constants_to_initializers(onnx.load(str(tmp)))
    tmp.unlink()
    return m


fe = torch.jit.load(str(src / "feature_extractor_cpu.pt"), map_location="cpu").eval()
m = export(torch.jit.script(Wrap(fe).eval()), torch.randn(1, 200, 160), dst / "sidon_fe.onnx", "x", "h", 1, 1)
onnx.save(m, str(dst / "_fe_fp32.onnx"))
quantize_dynamic(str(dst / "_fe_fp32.onnx"), str(dst / "sidon_fe_q8.onnx"), weight_type=QuantType.QUInt8,
                 per_channel=True, reduce_range=False, op_types_to_quantize=["MatMul"])
onnx.save(weights_to_fp16(m), str(dst / "sidon_fe.onnx"))
(dst / "_fe_fp32.onnx").unlink()

dec = torch.jit.load(str(src / "decoder_cpu.pt"), map_location="cpu").eval()
m = export(dec, torch.randn(1, 1024, 100), dst / "sidon_dec.onnx", "f", "y", 2, 2)
onnx.save(weights_to_fp16(m), str(dst / "sidon_dec.onnx"))
print("done")
