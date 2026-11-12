"""Stage 2: prepare INT8 weights, INT32 biases and per-layer scale settings.

Start at prepare_quantization(). Float models need representative calibration;
compatible Q/DQ models supply their own scales. Both return the same layer dicts.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from frontend import run_float
from isa import REGIONS


def prepare_quantization(model, calibration=None):
    """Return (quantized_layers, report); never recalibrate a Q/DQ graph."""
    if model.quantized_graph is not None:
        if calibration is not None:
            raise ValueError("Q/DQ model already contains quantization parameters: omit --calibration")
        return QDQImporter(model.quantized_graph).run()
    if calibration is None:
        candidates = (model.source.with_suffix('.calibration.npy'),
                      model.source.parent / 'calibration.npy')
        calibration = next((p for p in candidates if p.is_file()), None)
        if calibration is None:
            raise ValueError("Float-model INT8 compilation requires representative calibration inputs. "
                             "Use --calibration PATH.npy or a model.calibration.npy/calibration.npy sidecar.")
    samples = np.load(calibration, allow_pickle=False)
    layers = quantize(model.layers, samples)
    report = dict(mode='float-ptq', calibration_used=True, calibration=str(calibration),
                  source_input_dtype='float32', source_output_dtype='float32')
    return layers, report


# Float-model calibration and shared integer arithmetic.

def scale_for(x):
    peak = float(np.max(np.abs(x)))
    return peak / 127.0 if peak else 1.0


def int8(x, scale):
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) / scale), -128, 127).astype(np.int8)


def multiplier(ratio):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Requantization ratio must be finite and positive")
    for shift in range(62, -1, -1):
        m = round(math.ldexp(ratio, shift))
        if 0 < m <= 0x7fffffff:
            return m, shift
    raise ValueError("Requantization ratio is outside positive INT32 multiplier range")


def round_shift(value, shift):
    """Signed round-to-nearest, ties-to-even; no floating point at runtime."""
    if not 0 <= shift <= 62:
        raise ValueError("Shift out of range")
    if not shift:
        return int(value)
    sign = -1 if value < 0 else 1
    whole, rem = divmod(abs(int(value)), 1 << shift)
    half = 1 << (shift - 1)
    return sign * (whole + int(rem > half or (rem == half and whole & 1)))


def requantize(acc, m, shift, relu):
    x = max(0, int(acc)) if relu else int(acc)
    return max(-128, min(127, round_shift(x * int(m), shift)))


def quantize(layers, calibration):
    x = np.asarray(calibration, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != layers[0].weights.shape[1]:
        raise ValueError("Calibration must be nonempty [samples, input_features]")
    if not np.isfinite(x).all():
        raise ValueError("Calibration contains NaN/Inf")
    float_outputs = run_float(layers, x)
    if not all(np.isfinite(y).all() for y in float_outputs):
        raise ValueError("Non-finite calibration activations")
    input_scale = scale_for(x)
    result = []
    for layer, y in zip(layers, float_outputs):
        weight_scale, output_scale = scale_for(layer.weights), scale_for(y)
        acc_scale = input_scale * weight_scale
        if not math.isfinite(acc_scale) or acc_scale <= 0:
            raise ValueError("Accumulator scale is not representable")
        w = int8(layer.weights, weight_scale)
        b_float = np.rint(layer.bias / acc_scale)
        if not np.isfinite(b_float).all() or np.any(np.abs(b_float) > 0x7fffffff):
            raise ValueError(f"{layer.name}: bias cannot be represented in INT32")
        b = b_float.astype(np.int64)
        # Conservative bound covers every possible INT8 input and all partial sums.
        bound = np.abs(b) + 128 * np.abs(w.astype(np.int64)).sum(axis=1)
        if np.any(bound > 0x7fffffff):
            raise ValueError(f"{layer.name}: possible INT32 accumulator overflow")
        m, s = multiplier(acc_scale / output_scale)
        result.append(dict(name=layer.name, weights=w.tolist(), bias=b.tolist(),
                           activation=layer.activation, input_scale=input_scale,
                           weight_scale=weight_scale, output_scale=output_scale,
                           multiplier=m, shift=s))
        input_scale = output_scale
    return result


# Embedded Q/DQ import: checks stay explicit because silently changing a scale
# or a quantization boundary changes the meaning of the trained model.
@dataclass
class Constant:
    value: np.ndarray
    codes: object = None
    scale: object = None
    representation: str = "float"
    dequantized: bool = False

class QDQImporter:
    def __init__(self, model):
        from onnx import numpy_helper, TensorProto
        self.T = TensorProto
        self.g = model.graph
        versions = [x.version for x in model.opset_import if x.domain in ("", "ai.onnx")]
        if len(versions) != 1 or not 13 <= versions[0] <= 21:
            raise ValueError("Quantized import supports standard ONNX opsets 13..21")
        self.constants = {t.name: Constant(numpy_helper.to_array(t)) for t in self.g.initializer}
        self.nodes = {}
        self.users = {}
        self.used = set()
        self.report = []
        for n in self.g.node:
            if n.domain not in ("", "ai.onnx") or len(n.output) != 1:
                raise ValueError(f"Q/DQ import: unsupported domain or output count at {n.name}")
            if n.op_type in ("QLinearMatMul", "QLinearConv", "MatMulInteger", "DynamicQuantizeLinear"):
                raise ValueError("QOperator/dynamic quantization is unsupported; export static ONNX Q/DQ")
            self.nodes[n.output[0]] = n
            for x in set(n.input):
                self.users.setdefault(x, []).append(n)
        inputs = [x for x in self.g.input if x.name not in self.constants]
        if len(inputs) != 1 or len(self.g.output) != 1:
            raise ValueError("Q/DQ import needs one input and one output, with no runtime branches")
        self.input, self.output = inputs[0], self.g.output[0]
        for x in (self.input, self.output):
            t = x.type.tensor_type
            if t.elem_type not in (TensorProto.FLOAT, TensorProto.INT8) or len(t.shape.dim) != 2:
                raise ValueError("Q/DQ I/O must be rank-2 FLOAT or signed INT8 [batch,features]")

    @staticmethod
    def attrs(n):
        from onnx import helper
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def mark(self, n):
        self.used.add(n.output[0])

    def params(self, n, dtype):
        """Static, positive scalar scale and zero point 0 of the exact dtype."""
        a = self.attrs(n)
        if set(a) - {"axis"}:
            raise ValueError(f"{n.name or n.op_type}: unsupported quantization attributes {sorted(a)}")
        scale = self.const(n.input[1]).value
        if scale.dtype != np.float32 or scale.shape not in ((), (1,)):
            raise ValueError("Per-channel/blocked scales unsupported: expected one FLOAT32 scale per tensor")
        s = float(scale.reshape(-1)[0])
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Quantization scale must be finite and positive")
        if len(n.input) > 2 and n.input[2]:
            zp = self.const(n.input[2]).value
            # ORT emits scalar INT32 bias zero points with shape-[1] scales.
            # Both spellings represent one per-tensor value, not per-channel.
            if zp.shape not in ((), (1,)) or zp.dtype != dtype or np.any(zp != 0):
                raise ValueError(f"Expected scalar {np.dtype(dtype).name} zero point 0; "
                                 "unsigned/asymmetric/per-channel quantization is unsupported")
        elif n.op_type == "QuantizeLinear":
            raise ValueError("QuantizeLinear needs an explicit INT8 zero point 0 (ONNX default is UINT8)")
        return s

    def const(self, name):
        """Resolve only constants and supported constant transforms, retaining dtype."""
        from onnx import numpy_helper
        if name in self.constants:
            return self.constants[name]
        n = self.nodes.get(name)
        if n is None:
            raise ValueError(f"Expected constant tensor: {name}")
        a = self.attrs(n)
        op = n.op_type
        if op == "Constant" and set(a) == {"value"}:
            c = Constant(numpy_helper.to_array(a["value"]))
        elif op in ("Identity", "Transpose"):
            x = self.const(n.input[0])
            if set(a) - ({"perm"} if op == "Transpose" else set()):
                raise ValueError("Unsupported constant transform attributes")
            def transform(v):
                if v is None or op == "Identity":
                    return v
                return v.transpose(a.get("perm"))
            c = Constant(transform(x.value), transform(x.codes), x.scale,
                         x.representation, x.dequantized)
        elif op == "QuantizeLinear":
            x = self.const(n.input[0])
            s = self.params(n, np.dtype("int8"))
            if x.value.dtype != np.float32 or x.codes is not None or not np.isfinite(x.value).all():
                raise ValueError("Constant QAT QuantizeLinear must take finite FLOAT32 values")
            # ONNX Q executes a FLOAT32 division and nearest-even rounding.
            codes = np.clip(np.rint(x.value / np.float32(s)), -128, 127).astype(np.int8)
            c = Constant(codes, codes, s, "float-QDQ", False)
        elif op == "DequantizeLinear":
            x = self.const(n.input[0])
            if x.value.dtype not in (np.dtype("int8"), np.dtype("int32")) or x.dequantized:
                raise ValueError("Constant DQ needs signed INT8 weights or INT32 biases")
            s = self.params(n, x.value.dtype)
            if x.scale is not None and x.scale != s:
                raise ValueError("Constant Q and DQ parameters differ")
            c = Constant(x.value.astype(np.float32) * np.float32(s), x.value.copy(), s,
                         x.representation if x.scale is not None else "integer-DQ", True)
            if not np.isfinite(c.value).all():
                raise ValueError("Constant DQ values must remain finite in FLOAT32")
        else:
            raise ValueError(f"Unsupported constant expression {op} at {name}")
        self.mark(n)
        self.constants[name] = c
        return c

    def consumer(self, name):
        users = self.users.get(name, [])
        if name == self.output.name:
            if users:
                raise ValueError("Graph output is also consumed internally; branching unsupported")
            return None
        if len(users) != 1:
            raise ValueError(f"Expected one runtime consumer for {name}, got {len(users)}; branches unsupported")
        return users[0]

    def skip_identity(self, name):
        while True:
            n = self.consumer(name)
            if n is None or n.op_type != "Identity":
                return name
            self.mark(n)
            name = n.output[0]

    def dq_runtime(self, name, expected=None):
        name = self.skip_identity(name)
        n = self.consumer(name)
        if n is None or n.op_type != "DequantizeLinear" or n.input[0] != name:
            raise ValueError("Every quantized activation must feed a matching DequantizeLinear")
        s = self.params(n, np.dtype("int8"))
        if expected is not None and s != expected:
            raise ValueError("Activation Q/DQ scales differ; refusing to change tensor meaning")
        self.mark(n)
        return self.skip_identity(n.output[0]), s

    def bias(self, name, n, acc_scale):
        if not name:
            return np.zeros(n, dtype=np.int64), dict(bias_mode="zero", bias_max_abs_error=0.0)
        c = self.const(name)
        if c.value.shape not in ((), (n,), (1, n)):
            raise ValueError("Bias must be scalar, [N], or [1,N]")
        if c.dequantized:
            if c.codes.dtype != np.int32:
                raise ValueError("Quantized bias must be INT32")
            if not np.isclose(c.scale, acc_scale, rtol=2e-7, atol=0):
                raise ValueError("INT32 bias DQ scale must equal input_scale * weight_scale")
            b = np.broadcast_to(c.codes.reshape(-1), (n,)).astype(np.int64)
            error = float(np.max(np.abs(b.astype(np.float64) * (c.scale - acc_scale))))
            return b, dict(bias_mode="preserved-int32", bias_max_abs_error=error,
                          bias_declared_scale=c.scale)
        if c.value.dtype != np.float32 or not np.isfinite(c.value).all():
            raise ValueError("Unquantized bias must be finite FLOAT32")
        original = np.broadcast_to(c.value.reshape(-1), (n,)).astype(np.float64)
        rounded = np.rint(original / acc_scale)
        if not np.isfinite(rounded).all() or np.any(rounded < -2**31) or np.any(rounded > 2**31-1):
            raise ValueError("Imported float bias cannot fit INT32")
        return rounded.astype(np.int64), dict(bias_mode="float-rounded-to-int32",
            bias_max_abs_error=float(np.max(np.abs(rounded * acc_scale - original))))

    def run(self):
        current = self.skip_identity(self.input.name)
        if self.input.type.tensor_type.elem_type == self.T.FLOAT:
            q = self.consumer(current)
            if q is None or q.op_type != "QuantizeLinear" or q.input[0] != current:
                raise ValueError("Quantized import needs an input Q/DQ boundary; mixed float graph unsupported")
            sx = self.params(q, np.dtype("int8"))
            self.mark(q)
            current, sx = self.dq_runtime(q.output[0], sx)
        else:
            current, sx = self.dq_runtime(current)
        initial_width = self.input.type.tensor_type.shape.dim[1]
        width = initial_width.dim_value if initial_width.HasField("dim_value") else None
        initial_scale = sx
        layers = []
        while current != self.output.name:
            node = self.consumer(current)
            if node is None:
                break
            a = self.attrs(node)
            kind = node.op_type
            name = node.name or f"stage{len(layers)}"
            if kind in ("Gemm", "MatMul"):
                if node.input[0] != current:
                    raise ValueError("Dense runtime input must be the first operand")
                cw = self.const(node.input[1])
                if not cw.dequantized or cw.codes.dtype != np.int8 or cw.codes.ndim != 2:
                    raise ValueError("Dense weights must be constant signed INT8 DQ or FLOAT Q/DQ")
                if kind == "Gemm":
                    if (set(a) - {"alpha", "beta", "transA", "transB"} or a.get("transA", 0)
                        or a.get("transB", 0) not in (0, 1)
                        or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0):
                        raise ValueError("Quantized Gemm needs alpha=beta=1, transA=0 and transB=0/1")
                    w = cw.codes if a.get("transB", 0) else cw.codes.T
                    bname = node.input[2] if len(node.input) > 2 else ""
                else:
                    if a:
                        raise ValueError("Unexpected MatMul attributes")
                    w, bname = cw.codes.T, ""
                sw = cw.scale
                n, k = w.shape
                representation = cw.representation
                lowered = "dense"
                activation = "none"
            elif kind == "Relu":
                if width is None or width < 1:
                    raise ValueError("Standalone quantized ReLU requires a known feature width")
                # A separate Q boundary must not be erased: use an identity
                # dense stage to preserve its rounding/clipping with ISA v1.
                n = k = width
                if 16 * ((width + 3) // 4)**2 > REGIONS['weights'][1] - REGIONS['weights'][0]:
                    raise ValueError("Standalone ReLU identity lowering exceeds weight SRAM capacity")
                w, sw, bname = np.eye(width, dtype=np.int8), 1.0, ""
                representation, lowered, activation = "identity", "relu-identity-stage", "relu"
            else:
                raise ValueError(f"Unsupported quantized runtime op {kind}; expected Dense or ReLU")
            if min(n, k) < 1 or (width is not None and width != k):
                raise ValueError("Quantized dense dimensions disagree with preceding tensor")
            self.mark(node)
            current = self.skip_identity(node.output[0])
            candidate = self.consumer(current)
            if candidate is not None and candidate.op_type == "Add" and kind == "MatMul":
                if current not in candidate.input or len(candidate.input) != 2:
                    raise ValueError("Bias Add must use the preceding MatMul")
                bname = candidate.input[1] if candidate.input[0] == current else candidate.input[0]
                self.mark(candidate)
                current = self.skip_identity(candidate.output[0])
                candidate = self.consumer(current)
            if candidate is not None and candidate.op_type == "Relu" and kind != "Relu":
                activation = "relu"
                self.mark(candidate)
                current = self.skip_identity(candidate.output[0])
                candidate = self.consumer(current)
            if candidate is None or candidate.op_type != "QuantizeLinear" or candidate.input[0] != current:
                raise ValueError("Each dense/ReLU stage needs an output QuantizeLinear with a static scale")
            sy = self.params(candidate, np.dtype("int8"))
            self.mark(candidate)
            b, bias_report = self.bias(bname, n, sx*sw)
            bound = np.abs(b) + 128 * np.abs(w.astype(np.int64)).sum(axis=1)
            if np.any(bound > 2**31-1):
                raise ValueError(f"{name}: possible INT32 accumulator overflow")
            m, shift = multiplier(sx*sw/sy)
            layers.append(dict(name=name, weights=w.tolist(), bias=b.tolist(), activation=activation,
                               input_scale=sx, weight_scale=sw, output_scale=sy,
                               multiplier=m, shift=shift))
            self.report.append(dict(name=name, lowering=lowered, weight_representation=representation,
                input_scale=sx, weight_scale=sw, output_scale=sy,
                ratio=sx*sw/sy, target_ratio=m/(1<<shift),
                ratio_abs_error=abs(m/(1<<shift)-sx*sw/sy), **bias_report))
            width = n
            current = self.skip_identity(candidate.output[0])
            if current == self.output.name:
                if self.output.type.tensor_type.elem_type != self.T.INT8:
                    raise ValueError("QuantizeLinear graph output must be INT8")
                break
            current, sx = self.dq_runtime(current, sy)
            if current == self.output.name and self.output.type.tensor_type.elem_type != self.T.FLOAT:
                raise ValueError("DequantizeLinear graph output must be FLOAT")
        if not layers:
            raise ValueError("Quantized graph has no executable dense/ReLU stages")
        out_dim = self.output.type.tensor_type.shape.dim[1]
        if out_dim.HasField("dim_value") and out_dim.dim_value != width:
            raise ValueError("Quantized graph output width mismatch")
        unused = [n.op_type for n in self.g.node if n.output[0] not in self.used and n.op_type != "Constant"]
        if unused:
            raise ValueError(f"Unsupported/unused graph branches or operations: {unused}")
        return layers, dict(mode="imported-qdq", source_input_dtype="int8" if
            self.input.type.tensor_type.elem_type == self.T.INT8 else "float32",
            source_output_dtype="int8" if self.output.type.tensor_type.elem_type == self.T.INT8 else "float32",
            input_scale=initial_scale, layers=self.report,
            calibration_used=False, zero_point=0, granularity="per-tensor",
            note="Existing scales are preserved. Integer M/S approximation, bias rounding, and "
                 "FLOAT32 Q/DQ arithmetic can differ near rounding thresholds. Validate model accuracy.")
