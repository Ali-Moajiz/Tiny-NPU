"""Small standard ONNX Q/DQ fixtures, not newly trained PTQ/QAT networks.

PTQ-style stores INT8 weight constants. QAT-style stores FLOAT constants followed
by Q/DQ nodes. Both have identical quantized parameters. This demonstrates the
export contract without depending on any particular training framework.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def make_model(widths=(4, 4, 4), qat=False, int_io=False, matmul=False,
               float_bias=False, separated_relu=False, seed=77):
    rng = np.random.default_rng(seed)
    nodes, tensors, params = [], [], []

    def const(name, value):
        tensors.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def qnode(op, inp, scale_name, zero, out, **attrs):
        nodes.append(helper.make_node(op, [inp, scale_name, zero], [out], name=out, **attrs))
        return out

    zero = const('zero8', np.array(0, np.int8))
    zero32 = const('zero32', np.array(0, np.int32))
    sx = 0.125
    xs = const('input_scale', np.array(sx, np.float32))
    current = 'input' if int_io else qnode('QuantizeLinear', 'input', xs, zero, 'input_q')
    current = qnode('DequantizeLinear', current, xs, zero, 'input_dq')
    for i, (k, n) in enumerate(zip(widths, widths[1:])):
        w = rng.integers(-12, 13, (n, k), dtype=np.int8)
        b = rng.integers(-30, 31, n, dtype=np.int32)
        sw, sy = 0.0625, 0.25 * 2**i
        ws = const(f'w{i}_scale', np.array(sw, np.float32))
        # Gemm transB=1 consumes [N,K]; MatMul consumes [K,N].
        stored = w.T.copy() if matmul else w
        if qat:
            raw = const(f'w{i}_float', stored.astype(np.float32)*np.float32(sw))
            raw = qnode('QuantizeLinear', raw, ws, zero, f'w{i}_q')
        else:
            raw = const(f'w{i}_int8', stored)
        wdq = qnode('DequantizeLinear', raw, ws, zero, f'w{i}_dq')
        if float_bias:
            bias = const(f'b{i}_float', b.astype(np.float32)*np.float32(sx*sw))
        else:
            bs = const(f'b{i}_scale', np.array(sx*sw, np.float32))
            bi = const(f'b{i}_int32', b)
            bias = qnode('DequantizeLinear', bi, bs, zero32, f'b{i}_dq')
        if matmul:
            nodes.append(helper.make_node('MatMul', [current, wdq], [f'mat{i}'], name=f'fc{i}'))
            nodes.append(helper.make_node('Add', [f'mat{i}', bias], [f'z{i}'], name=f'bias{i}'))
        else:
            nodes.append(helper.make_node('Gemm', [current, wdq, bias], [f'z{i}'], name=f'fc{i}', transB=1))
        current = f'z{i}'
        relu = i < len(widths)-2
        if relu and not separated_relu:
            nodes.append(helper.make_node('Relu', [current], [f'relu{i}'], name=f'relu{i}'))
            current = f'relu{i}'
        ys = const(f'y{i}_scale', np.array(sy, np.float32))
        current = qnode('QuantizeLinear', current, ys, zero, f'y{i}_q')
        params.append(dict(weights=w, bias=b, input_scale=sx, weight_scale=sw, output_scale=sy))
        if i == len(widths)-2 and int_io:
            break
        current = qnode('DequantizeLinear', current, ys, zero, f'y{i}_dq')
        if relu and separated_relu:
            nodes.append(helper.make_node('Relu', [current], [f'relu{i}'], name=f'relu{i}'))
            sy *= 2
            ys = const(f'relu{i}_scale', np.array(sy, np.float32))
            current = qnode('QuantizeLinear', f'relu{i}', ys, zero, f'relu{i}_q')
            current = qnode('DequantizeLinear', current, ys, zero, f'relu{i}_dq')
        sx = sy
    graph = helper.make_graph(nodes, 'qat_style' if qat else 'ptq_style',
        [helper.make_tensor_value_info('input', TensorProto.INT8 if int_io else TensorProto.FLOAT, [None, widths[0]])],
        [helper.make_tensor_value_info(current, TensorProto.INT8 if int_io else TensorProto.FLOAT, [None, widths[-1]])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 19)], ir_version=9)
    onnx.checker.check_model(model)
    return model, params


def main():
    root = Path(__file__).resolve().parent
    for name, options in [('ptq_qdq.onnx', {}), ('qat_qdq.onnx', {'qat': True}),
                          ('ptq_qdq_int8_io.onnx', {'int_io': True})]:
        model, _ = make_model(**options)
        onnx.save(model, root/name)
        print(root/name)


if __name__ == '__main__':
    main()
