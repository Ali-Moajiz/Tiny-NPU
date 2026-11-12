"""Stage 1: load a model. Start at load_model(). No NPU instructions here.

Model.layers holds normalized floating-point dense layers. For Q/DQ ONNX,
Model.quantized_graph holds the checked graph for quantization.py to interpret.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class Model:
    source: Path
    layers: list | None = None
    quantized_graph: object | None = None


@dataclass
class Layer:
    name: str
    weights: np.ndarray  # [output neuron, input feature]
    bias: np.ndarray     # [output neuron]
    activation: str = "none"

def validate(layers):
    if not layers:
        raise ValueError("Model must contain at least one dense layer")
    previous = None
    for layer in layers:
        w, b = layer.weights, layer.bias
        if w.ndim != 2 or min(w.shape) < 1 or b.shape != (w.shape[0],):
            raise ValueError(f"{layer.name}: expected weights [N,K], bias [N]")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError(f"{layer.name}: non-finite parameters")
        if layer.activation not in ("none", "relu"):
            raise ValueError(f"{layer.name}: unsupported activation {layer.activation!r}")
        if previous is not None and previous != w.shape[1]:
            raise ValueError(f"{layer.name}: incompatible adjacent layer dimensions")
        previous = w.shape[0]
    return layers

def load_json(path):
    data = json.loads(Path(path).read_text())
    if data.get("format") != "tiny-npu-float-v1":
        raise ValueError("JSON model format must be tiny-npu-float-v1")
    return validate([Layer(x.get("name", f"layer{i}"),
                           np.asarray(x["weights"], dtype=np.float64),
                           np.asarray(x["bias"], dtype=np.float64),
                           x.get("activation", "none"))
                     for i, x in enumerate(data["layers"])])

def run_float(layers, inputs):
    x = np.asarray(inputs, dtype=np.float64)
    outputs = []
    for layer in layers:
        x = x @ layer.weights.T + layer.bias
        if layer.activation == "relu":
            x = np.maximum(x, 0)
        outputs.append(x.copy())
    return outputs


def load_model(path):
    """Read ONNX (or legacy float JSON) without writing intermediate files."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Model file does not exist: {path}")
    if path.suffix.lower() == ".json":
        return Model(path, layers=load_json(path))
    if path.suffix.lower() != ".onnx":
        raise ValueError("Input must be .onnx or legacy tiny-npu-float-v1 .json")
    import onnx
    graph = onnx.load(str(path), load_external_data=True)
    try:
        onnx.checker.check_model(graph)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"Invalid ONNX model: {exc}") from exc
    quantized = any(node.op_type in ("QuantizeLinear", "DequantizeLinear",
                    "DynamicQuantizeLinear", "MatMulInteger", "ConvInteger")
                    or node.op_type.startswith("QLinear")
                    or "quant" in node.op_type.lower() for node in graph.graph.node)
    if quantized:
        return Model(path, quantized_graph=graph)
    return Model(path, layers=read_float_layers(graph))


def read_float_layers(model):
    """Normalize a sequential Gemm/MatMul/Add/ReLU graph to Layer objects."""
    import onnx
    from onnx import helper, numpy_helper
    versions = [x.version for x in model.opset_import if x.domain in ("", "ai.onnx")]
    if len(versions) != 1 or not 7 <= versions[0] <= 21:
        raise ValueError("Supported standard ONNX opsets: 7 through 21; export opset 17")
    g = model.graph
    constants = {t.name: numpy_helper.to_array(t).astype(np.float64)
                 for t in g.initializer}
    inputs = [x for x in g.input if x.name not in constants]
    if len(inputs) != 1 or len(g.output) != 1:
        raise ValueError("Exactly one runtime input and one output are supported")
    for tensor in (inputs[0], g.output[0]):
        tt = tensor.type.tensor_type
        if tt.elem_type != onnx.TensorProto.FLOAT or len(tt.shape.dim) != 2:
            raise ValueError("Expected rank-2 FLOAT tensors [batch, features]")
    current = inputs[0].name
    layers = []
    can_add_bias = False
    # Follow the single runtime chain in topological order. Constant transforms
    # commonly emitted by exporters do not consume a runtime value.
    for node in g.node:
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1:
            raise ValueError(f"Unsupported domain or output count at {node.name}")
        a = {v.name: helper.get_attribute_value(v) for v in node.attribute}
        op = node.op_type
        out = node.output[0]
        if op == "Constant":
            if set(a) != {"value"}:
                raise ValueError("Only tensor-valued Constant is supported")
            constants[out] = numpy_helper.to_array(a["value"]).astype(np.float64)
            continue
        if op in ("Transpose", "Identity") and node.input[0] in constants:
            c = constants[node.input[0]]
            constants[out] = c.T if op == "Transpose" and "perm" not in a else (
                c.transpose(a["perm"]) if op == "Transpose" else c.copy())
            continue
        if op == "Identity" and list(node.input) == [current]:
            current = out
            continue
        if op in ("Gemm", "MatMul"):
            if node.input[0] != current or node.input[1] not in constants:
                raise ValueError("Dense input must be the current chain; weights must be constant")
            w = constants[node.input[1]]
            if w.ndim != 2:
                raise ValueError("Dense weights must have rank 2")
            if op == "Gemm":
                if set(a) - {"alpha", "beta", "transA", "transB"} or a.get("transA", 0):
                    raise ValueError("Unsupported Gemm attributes/transA")
                if a.get("transB", 0) not in (0, 1):
                    raise ValueError("Invalid Gemm transB")
                w = w if a.get("transB", 0) else w.T
                w = w * float(a.get("alpha", 1))
                if len(node.input) > 2 and node.input[2]:
                    if node.input[2] not in constants:
                        raise ValueError("Gemm bias must be constant")
                    b = constants[node.input[2]] * float(a.get("beta", 1))
                else:
                    b = np.zeros(w.shape[0])
                can_add_bias = False
            else:
                if a:
                    raise ValueError("Unexpected MatMul attributes")
                w = w.T
                b = np.zeros(w.shape[0])
                can_add_bias = True
            if b.shape not in ((), (w.shape[0],), (1, w.shape[0])):
                raise ValueError("Bias must be scalar, [N], or [1,N]")
            b = np.broadcast_to(b.reshape(-1), (w.shape[0],)).copy()
            layers.append(Layer(node.name or f"layer{len(layers)}", w.copy(), b))
        elif op == "Add" and can_add_bias:
            if len(node.input) != 2 or current not in node.input:
                raise ValueError("Add must apply constant bias to the preceding MatMul")
            other = node.input[1] if node.input[0] == current else node.input[0]
            if other not in constants:
                raise ValueError("Residual/dynamic Add is not supported")
            b = constants[other]
            n = layers[-1].weights.shape[0]
            if b.shape not in ((), (n,), (1, n)):
                raise ValueError("Add bias must be scalar, [N], or [1,N]")
            layers[-1].bias = np.broadcast_to(b.reshape(-1), (n,)).copy()
            can_add_bias = False
        elif op == "Relu" and layers and list(node.input) == [current]:
            layers[-1].activation = "relu"
            can_add_bias = False
        else:
            raise ValueError(f"Unsupported node/branch {node.name!r}: {op}")
        current = out
    if current != g.output[0].name:
        raise ValueError("Graph output must be the end of the supported dense chain")
    validate(layers)
    in_dim = inputs[0].type.tensor_type.shape.dim[1]
    if in_dim.HasField("dim_value") and in_dim.dim_value != layers[0].weights.shape[1]:
        raise ValueError("Graph input width differs from dense weight shape")
    out_dim = g.output[0].type.tensor_type.shape.dim[1]
    if out_dim.HasField("dim_value") and out_dim.dim_value != layers[-1].weights.shape[0]:
        raise ValueError("Graph output width differs from dense weight shape")
    return layers
