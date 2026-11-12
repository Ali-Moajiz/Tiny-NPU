"""Regression tests; test models/data are generated in temporary storage.

Run: python3 -m unittest test_compiler -v
Only the optional real-export test needs onnxruntime. No test fixtures are
installed in compiler/, and no outputs are left in the project by these tests.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from fractions import Fraction
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator

from main import main, compile_file
from frontend import Layer, run_float, load_model as read_model
from quantization import QDQImporter as Importer, quantize, int8, round_shift, requantize
from compiler import compile_model as schedule_model
from isa import Instruction, Op, ENTRY_PC, SRAM_BYTES, REGIONS, address, dimensions, disassemble, encode_program
from verify import execute, pack_tensor, unpack_tensor, oracle, run_directory


# Helpers let the pre-refactor numerical tests exercise the new staged API.
def load_model(path):
    return read_model(path).layers


def compile_model(model, calibration, output, batch=4, forward=True):
    return compile_file(model, calibration, output, batch, forward, debug=True)[0]


def build_quantized(layers, batch=4, forward=True):
    compilation = schedule_model(layers, batch, forward)
    program = encode_program(compilation.instructions)
    return compilation.image(program), program, compilation.manifest, layers, compilation.schedule


def build(layers, calibration, batch=4, forward=True):
    return build_quantized(quantize(layers, calibration), batch, forward)


def write_float_fixture(target):
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 0.7, (4, 4)).astype(np.float32)
    w2 = rng.normal(0, 0.7, (4, 4)).astype(np.float32)
    b1 = rng.normal(0, 0.2, 4).astype(np.float32)
    b2 = rng.normal(0, 0.2, 4).astype(np.float32)
    layers = [dict(name="fc1", weights=w1.tolist(), bias=b1.tolist(), activation="relu"),
              dict(name="fc2", weights=w2.tolist(), bias=b2.tolist(), activation="none")]
    (target / "model.json").write_text(json.dumps(dict(format="tiny-npu-float-v1", layers=layers), indent=2)+"\n")
    nodes = [helper.make_node("Gemm", ["input", "W1", "B1"], ["z1"], name="fc1", transB=1),
             helper.make_node("Relu", ["z1"], ["h1"], name="relu"),
             helper.make_node("Gemm", ["h1", "W2", "B2"], ["output"], name="fc2", transB=1)]
    tensors = [numpy_helper.from_array(x, name) for x, name in ((w1,"W1"),(b1,"B1"),(w2,"W2"),(b2,"B2"))]
    graph = helper.make_graph(nodes, "example444",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, target / "model.onnx")
    np.save(target / "calibration.npy", rng.uniform(-4, 4, (256, 4)).astype(np.float32))
    np.save(target / "inputs.npy", np.array([[1,2,3,4], [0,1,-4,-1], [0,0,0,0], [-1,2,-3,4]], dtype=np.float32))


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

_fixtures = None
EXAMPLES = None


def setUpModule():
    global _fixtures, EXAMPLES
    _fixtures = tempfile.TemporaryDirectory(prefix='tiny-npu-tests-')
    EXAMPLES = Path(_fixtures.name)
    write_float_fixture(EXAMPLES)


def tearDownModule():
    _fixtures.cleanup()


def save_json(path, layers):
    path.write_text(json.dumps(dict(format="tiny-npu-float-v1", layers=[
        dict(name=l.name, weights=l.weights.tolist(), bias=l.bias.tolist(), activation=l.activation)
        for l in layers])))

def program_image(instructions):
    image = bytearray(SRAM_BYTES)
    data = b"".join(x.encode() for x in instructions)
    image[ENTRY_PC:ENTRY_PC+len(data)] = data
    return image

class EncodingTests(unittest.TestCase):
    def test_golden_encoding_and_endianness(self):
        i = Instruction(Op.LOAD_OPERAND, src=REGIONS['activations'][0]+16, dest=4, dims=16)
        # Construct expected bit-string independently, field by field.
        bits = "1000" + f"{REGIONS['activations'][0]+16:032b}" + f"{4:032b}" + f"{16:028b}" + "0"*32
        golden = int(bits, 2)
        self.assertEqual(i.word(), golden)
        self.assertEqual(i.encode(), golden.to_bytes(16, "little"))
        self.assertEqual(Instruction.decode(i.encode()), i)
        self.assertEqual(Instruction(Op.HALT).encode(), b"\0"*15+b"\xf0")

    def test_all_opcodes_roundtrip(self):
        examples = [Instruction(Op.STREAM_LOAD, dest=REGIONS["activations"][0], dims=16),
                    Instruction(Op.EXEC_MAC, dims=dimensions(4, 3, 2), flags=3),
                    Instruction(Op.POST_PROC, dest=(REGIONS["activations"][0]+0x40), dims=dimensions(4, 3), flags=5),
                    Instruction(Op.FORWARD, dest=1, dims=dimensions(4, 3)),
                    Instruction(Op.STREAM_STORE, src=(REGIONS["activations"][0]+0x40), dims=16),
                    Instruction(Op.LOOP_START, src=4, dest=(ENTRY_PC+16)),
                    Instruction(Op.LOOP_END, dest=(ENTRY_PC+16)),
                    Instruction(Op.LOAD_OPERAND, src=REGIONS["bias"][0], dest=2, dims=16),
                    Instruction(Op.HALT)]
        for ins in examples:
            self.assertEqual(Instruction.decode(ins.encode()), ins)
        self.assertEqual({i.op for i in examples}, set(Op))

    def test_bad_instructions(self):
        cases = [Instruction(Op.LOAD_OPERAND, src=(REGIONS["activations"][0]+1), dims=16),
                 Instruction(Op.LOAD_OPERAND, src=REGIONS["activations"][0], dest=1, dims=16),
                 Instruction(Op.EXEC_MAC, dims=dimensions(5, 4, 4)),
                 Instruction(Op.EXEC_MAC, dims=dimensions(4, 4, 0)),
                 Instruction(Op.HALT, flags=1),
                 Instruction(Op.STREAM_LOAD, dest=REGIONS["activations"][0], dims=32)]
        for ins in cases:
            with self.subTest(ins=ins), self.assertRaises(ValueError):
                ins.encode()
        with self.assertRaises(ValueError):
            Instruction.decode(bytes(16))
        with self.assertRaises(ValueError):
            disassemble(bytes(17))

    def test_single_loop(self):
        code = [Instruction(Op.LOOP_START, src=3, dest=(ENTRY_PC+16)),
                Instruction(Op.STREAM_LOAD, dest=REGIONS["activations"][0], dims=16),
                Instruction(Op.STREAM_STORE, src=REGIONS["activations"][0], dims=16),
                Instruction(Op.LOOP_END, dest=(ENTRY_PC+16)), Instruction(Op.HALT)]
        data = bytes(range(48))
        result = execute(program_image(code), data)
        self.assertEqual(result["output"], data)
        self.assertEqual(result["instructions"], 11)

class NumericTests(unittest.TestCase):
    def test_signed_rounding_and_clamp(self):
        for shift in range(0, 12):
            for value in range(-200, 201):
                self.assertEqual(round_shift(value, shift), round(Fraction(value, 1 << shift)))
        self.assertEqual(requantize(-1000, 1, 0, False), -128)
        self.assertEqual(requantize(1000, 1, 0, False), 127)
        self.assertEqual(requantize(-1000, 1, 0, True), 0)

    def test_quantization_bias_units_zero_and_overflow(self):
        zero = Layer("zero", np.zeros((4, 4)), np.zeros(4))
        q = quantize([zero], np.zeros((5, 4)))[0]
        self.assertEqual(q["input_scale"], 1)
        self.assertEqual(q["bias"], [0]*4)
        bad = Layer("bad", np.ones((4, 4)), np.full(4, 1e20))
        with self.assertRaisesRegex(ValueError, "INT32"):
            quantize([bad], np.ones((4, 4)))
        with self.assertRaises(ValueError):
            quantize([zero], np.full((4, 4), np.nan))

    def test_tensor_layout(self):
        x = np.arange(21, dtype=np.int8).reshape(3, 7)
        beats = pack_tensor(x)
        self.assertEqual(len(beats), 32)
        self.assertEqual(list(beats[:8]), [0,1,2,3,7,8,9,10])
        self.assertEqual(list(beats[16:20]), [4,5,6,0])
        np.testing.assert_array_equal(unpack_tensor(beats, 3, 7), x)

class FrontendTests(unittest.TestCase):
    def test_example_onnx_against_onnx_reference(self):
        path = EXAMPLES / "model.onnx"
        layers = load_model(path)
        x = np.load(EXAMPLES / "inputs.npy")
        y = ReferenceEvaluator(str(path)).run(None, {"input": x})[0]
        np.testing.assert_allclose(run_float(layers, x)[-1], y, rtol=1e-5, atol=1e-6)
        j = load_model(EXAMPLES / "model.json")
        for a, b in zip(layers, j):
            np.testing.assert_array_equal(a.weights, b.weights)
            np.testing.assert_array_equal(a.bias, b.bias)

    def test_matmul_add_and_gemm_attributes(self):
        rng = np.random.default_rng(123)
        w, b = rng.normal(size=(3, 5)).astype(np.float32), rng.normal(size=(5,)).astype(np.float32)
        for kind in ("matmul", "gemm"):
            if kind == "matmul":
                nodes = [helper.make_node("MatMul", ["x", "w"], ["z"]),
                         helper.make_node("Add", ["b", "z"], ["a"]),
                         helper.make_node("Relu", ["a"], ["y"])]
            else:
                nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["a"], alpha=0.75, beta=1.25),
                         helper.make_node("Relu", ["a"], ["y"])]
            graph = helper.make_graph(nodes, kind,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 5])],
                [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)], ir_version=9)
            with tempfile.TemporaryDirectory() as t:
                path = Path(t)/"test.onnx"
                onnx.save(model, path)
                x = rng.normal(size=(4, 3)).astype(np.float32)
                y = ReferenceEvaluator(model).run(None, {"x": x})[0]
                np.testing.assert_allclose(run_float(load_model(path), x)[-1], y, rtol=1e-5, atol=1e-6)

    def test_unsupported_operator_rejected(self):
        model = onnx.load(EXAMPLES / "model.onnx")
        model.graph.node[1].op_type = "Sigmoid"
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/"bad.onnx"
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_model(path)

class EndToEndTests(unittest.TestCase):
    def test_tiling_all_batches_forward_modes_and_partial_batches(self):
        rng = np.random.default_rng(1234)
        shapes = [(4,4,4), (7,5,3), (5,9,2), (4,3,7), (1,1), (8,8), (9,3,2)]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for widths in shapes:
                layers = [Layer(f"L{i}", rng.normal(0, .4, (n,k)), rng.normal(0,.1,n),
                                "relu" if i < len(widths)-2 else "none")
                          for i, (k,n) in enumerate(zip(widths, widths[1:]))]
                save_json(root/"m.json", layers)
                np.save(root/"cal.npy", rng.normal(0,2,(32,widths[0])))
                x = rng.integers(-128,128,(9,widths[0]),dtype=np.int16)
                x[0] = -128
                x[1] = 127
                for batch in range(1,5):
                    for forward in (False, True):
                        with self.subTest(widths=widths,batch=batch,forward=forward):
                            m = compile_model(root/"m.json",root/"cal.npy",root/"out",batch,forward)
                            result = run_directory(root/"out", x, quantized=True)
                            self.assertEqual(len(result["output_int8"]), 9)
                            self.assertEqual(result["runs"][0]["useful_macs"], m["useful_macs_per_inference"])
                            data = (root/"out/sram.bin").read_bytes()
                            hexdata = b"".join(int(line,16).to_bytes(16,"little")
                                              for line in (root/"out/sram.hex").read_text().splitlines())
                            self.assertEqual(data, hexdata)

    def test_deterministic_build(self):
        layers = load_model(EXAMPLES/"model.onnx")
        cal = np.load(EXAMPLES/"calibration.npy")
        self.assertEqual(build(layers, cal), build(layers, cal))

    def test_program_capacity(self):
        layers = [Layer(f"L{i}", np.ones((8,8)), np.zeros(8)) for i in range(256)]
        with self.assertRaisesRegex(ValueError, "program capacity"):
            build(layers, np.ones((8,8)), forward=False)

    def test_parameter_capacity(self):
        layer = Layer("big", np.ones((900,900)), np.zeros(900))
        with self.assertRaisesRegex(ValueError, "weights capacity"):
            build([layer], np.ones((2,900)))

    def test_binary_faults(self):
        layers = load_model(EXAMPLES/"model.onnx")
        image, program, manifest, _, _ = build(layers, np.load(EXAMPLES/"calibration.npy"))
        inputs = bytes(16)
        bad = bytearray(image)
        bad[ENTRY_PC+15] &= 0x0f  # opcode zero
        with self.assertRaises(ValueError):
            execute(bad, inputs)
        bad = bytearray(image)
        cfg = manifest["layers"][0]["config"]
        bad[cfg:cfg+4] = bytes(4)  # multiplier zero
        with self.assertRaisesRegex(ValueError, "configuration"):
            execute(bad, inputs)
        bad = bytearray(image)
        bias = manifest["layers"][0]["bias"][0]
        bad[bias:bias+16] = struct.pack("<iiii", *([2**31-1]*4))
        weights = manifest["layers"][0]["weights"][0][0]
        bad[weights:weights+16] = bytes([127]*16)
        with self.assertRaisesRegex(ValueError, "overflow"):
            execute(bad, bytes([127]*16))
        invalid = program_image([Instruction(Op.EXEC_MAC,dims=dimensions(4,4,4)),Instruction(Op.HALT)])
        with self.assertRaisesRegex(ValueError, "invalid operands"):
            execute(invalid, b"")
        with self.assertRaisesRegex(ValueError, "underflow"):
            execute(image,b"")
        with self.assertRaisesRegex(ValueError, "unused"):
            execute(image,bytes(32))

class CommandTests(unittest.TestCase):
    def invoke(self, args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = main(args)
        return result, stream.getvalue()

    def test_one_argument_compile_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(EXAMPLES / "model.onnx", root / "network.onnx")
            shutil.copy(EXAMPLES / "calibration.npy", root / "network.calibration.npy")
            result, msg = self.invoke([str(root / "network.onnx")])
            self.assertEqual(result, 0, msg)
            self.assertEqual((root / "network_compiled/program.bin").stat().st_size, 240)
            self.assertIn("network.calibration.npy", msg)

    def test_generic_calibration_fallback_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "network.onnx"
            shutil.copy(EXAMPLES / "model.onnx", model)
            shutil.copy(EXAMPLES / "calibration.npy", root / "calibration.npy")
            self.assertEqual(self.invoke([str(model)])[0], 0)
            # Invalid model-specific file must win over the generic fallback.
            np.save(root / "network.calibration.npy", np.zeros((2, 7)))
            self.assertEqual(self.invoke([str(model)])[0], 2)
            result, msg = self.invoke([str(model), "--calibration", str(root / "calibration.npy"),
                                       "--out", str(root / "explicit"), "--batch", "1"])
            self.assertEqual(result, 0, msg)
            self.assertTrue((root / "explicit/sram.bin").is_file())

    def test_no_calibration_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(EXAMPLES / "model.onnx", root / "network.onnx")
            result, msg = self.invoke([str(root / "network.onnx")])
            self.assertEqual(result, 2)
            self.assertIn("requires representative calibration", msg)
            self.assertFalse((root / "network_compiled").exists())

    def test_missing_model(self):
        result, msg = self.invoke(["/no/such/model.onnx"])
        self.assertEqual(result, 2)
        self.assertIn("Model file does not exist", msg)

def replace_constant(model, name, value):
    for i, t in enumerate(model.graph.initializer):
        if t.name == name:
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(np.asarray(value), name))
            return
    raise AssertionError(name)

class ImportTests(unittest.TestCase):
    def test_ptq_and_qat_same_integer_program(self):
        ptq, p = make_model()
        qat, _ = make_model(qat=True)
        a, report_a = Importer(ptq).run()
        b, report_b = Importer(qat).run()
        self.assertEqual(a, b)
        self.assertEqual(build_quantized(a)[0:2], build_quantized(b)[0:2])
        self.assertEqual(report_a['layers'][0]['weight_representation'], 'integer-DQ')
        self.assertEqual(report_b['layers'][0]['weight_representation'], 'float-QDQ')
        np.testing.assert_array_equal(a[0]['weights'], p[0]['weights'])
        np.testing.assert_array_equal(a[0]['bias'], p[0]['bias'])
        self.assertEqual(a[0]['input_scale'], p[0]['input_scale'])
        self.assertEqual(a[0]['output_scale'], p[0]['output_scale'])

    def test_source_onnx_outputs_all_variants(self):
        rng = np.random.default_rng(998)
        # 3 shapes x 2 weight styles x 2 I/O types x 2 dense forms x 4 batches.
        for shape in [(4,4,4), (7,5,3), (4,3,7)]:
            for qat in [False, True]:
                for int_io in [False, True]:
                    for matmul in [False, True]:
                        model, _ = make_model(shape, qat=qat, int_io=int_io, matmul=matmul)
                        qlayers, _ = Importer(model).run()
                        ref = ReferenceEvaluator(model)
                        for batch in range(1,5):
                            with self.subTest(shape=shape,qat=qat,int_io=int_io,matmul=matmul,batch=batch):
                                xq = rng.integers(-128,128,(batch,shape[0]),dtype=np.int16).astype(np.int8)
                                x = xq if int_io else xq.astype(np.float32)*np.float32(qlayers[0]['input_scale'])
                                expected = ref.run(None, {'input':x})[0]
                                if not int_io:
                                    expected = np.rint(expected / qlayers[-1]['output_scale']).astype(np.int8)
                                for forward in (False, True):
                                    image, _, _, _, _ = build_quantized(qlayers, batch, forward)
                                    actual = unpack_tensor(execute(image,pack_tensor(xq))['output'],batch,shape[-1])
                                    np.testing.assert_array_equal(actual,expected)

    def test_cli_no_calibration_and_no_recalibration(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            model,_=make_model(qat=True)
            onnx.save(model,root/'net.onnx')
            # Even an invalid sidecar must not be opened for a quantized model.
            (root/'net.calibration.npy').write_bytes(b'not a numpy file')
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main([str(root/'net.onnx'), '--debug']),0)
            self.assertIn('imported-qdq',out.getvalue())
            manifest=json.loads((root/'net_compiled/manifest.json').read_text())
            self.assertIsNone(manifest['calibration'])
            self.assertEqual(manifest['quantization_mode'],'imported-qdq')
            checked=run_directory(root/'net_compiled',np.ones((5,4),dtype=np.float32))
            self.assertEqual(len(checked['output_int8']),5)
            with self.assertRaisesRegex(ValueError,'omit --calibration'):
                compile_model(root/'net.onnx',root/'net.calibration.npy',root/'bad')

    def test_separate_relu_quantization_boundary(self):
        for qat in [False,True]:
            model,_=make_model(qat=qat,separated_relu=True)
            qs,report=Importer(model).run()
            self.assertEqual(len(qs),3)
            self.assertEqual(report['layers'][1]['lowering'],'relu-identity-stage')
            image,*_=build_quantized(qs)
            x=np.array([[-128,127,-44,17],[1,2,3,4],[0,0,0,0],[-5,-7,9,12]],np.int8)
            expected=ReferenceEvaluator(model).run(None,{'input':x.astype(np.float32)*.125})[0]
            expected=np.rint(expected/qs[-1]['output_scale']).astype(np.int8)
            actual=unpack_tensor(execute(image,pack_tensor(x))['output'],4,4)
            np.testing.assert_array_equal(actual,expected)

    def test_float_bias_and_fractional_qat_weights(self):
        model,_=make_model(qat=True,float_bias=True)
        for t in model.graph.initializer:
            if t.name=='w0_float':
                values=numpy_helper.to_array(t).copy()
                # Half-way cases test FLOAT32 nearest-even Q, including signs.
                values.flat[:4]=np.array([.5,1.5,-.5,-1.5],np.float32)*.0625
                replace_constant(model,'w0_float',values)
                break
        qs,report=Importer(model).run()
        self.assertEqual(qs[0]['weights'][0],[0,2,0,-2])
        self.assertEqual(report['layers'][0]['bias_mode'],'float-rounded-to-int32')
        self.assertEqual(report['layers'][0]['bias_max_abs_error'],0)

    def test_nonexact_float_bias_error_report(self):
        model,_=make_model(qat=True,float_bias=True)
        replace_constant(model,'b0_float',np.array([.001,.002,.003,.004],np.float32))
        _,report=Importer(model).run()
        self.assertGreater(report['layers'][0]['bias_max_abs_error'],0)
        self.assertLessEqual(report['layers'][0]['bias_max_abs_error'],.125*.0625/2)

    def test_reject_unsupported_quantization(self):
        cases=[('zero8',np.array(1,np.int8),'zero point'),
               ('zero8',np.array(0,np.uint8),'zero point'),
               ('w0_scale',np.array([.1,.2,.3,.4],np.float32),'Per-channel'),
               ('input_scale',np.array(0,np.float32),'positive'),
               ('y0_scale',np.array(np.nan,np.float32),'positive'),
               ('b0_scale',np.array(.1,np.float32),'bias DQ scale')]
        for name,value,message in cases:
            with self.subTest(name=name,value=value):
                model,_=make_model()
                replace_constant(model,name,value)
                with self.assertRaisesRegex(ValueError,message):
                    Importer(model).run()

    def test_reject_mismatched_qdq_and_gemm_alpha(self):
        model,_=make_model(qat=True)
        model.graph.initializer.append(numpy_helper.from_array(np.array(.1,np.float32),'different'))
        for node in model.graph.node:
            if node.output[0]=='w0_dq':node.input[1]='different'
        with self.assertRaisesRegex(ValueError,'parameters differ'):
            Importer(model).run()
        model,_=make_model()
        for node in model.graph.node:
            if node.op_type=='Gemm':node.attribute.append(helper.make_attribute('alpha',.5));break
        with self.assertRaisesRegex(ValueError,'alpha=beta'):
            Importer(model).run()

    def test_reject_missing_boundary_and_branches(self):
        model,_=make_model()
        model.graph.node.append(helper.make_node('Identity',['input_dq'],['branch']))
        with self.assertRaisesRegex(ValueError,'branches'):
            Importer(model).run()
        model,_=make_model()
        # Bypass input quantization. Must not silently perform PTQ again.
        for node in model.graph.node:
            if node.op_type=='Gemm':node.input[0]='input';break
        with self.assertRaises(ValueError):
            Importer(model).run()

    def test_constant_transpose_and_identity(self):
        model,_=make_model()
        for node in model.graph.node:
            if node.name=='fc0':
                node.input[1]='transposed'
                for a in node.attribute:
                    if a.name=='transB':a.i=0
                break
        # Insert the constant transpose before its Gemm consumer.
        idx=next(i for i,n in enumerate(model.graph.node) if n.name=='fc0')
        model.graph.node.insert(idx,helper.make_node('Transpose',['w0_dq'],['transposed'],perm=[1,0]))
        actual,_=Importer(model).run()
        original,_=Importer(make_model()[0]).run()
        self.assertEqual(actual,original)

    def test_external_tensors_opsets_and_scalar_spellings(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            model,_=make_model()
            replace_constant(model,'b0_scale',np.array([.125*.0625],np.float32))
            for version in (13,17,19,21):
                model.opset_import[0].version=version
                path=root/f'net{version}.onnx'
                onnx.save_model(model,path,save_as_external_data=True,all_tensors_to_one_file=True,
                                location=f'net{version}.data',size_threshold=0)
                compiled=compile_model(path,None,root/f'out{version}')
                self.assertEqual(compiled['quantization_mode'],'imported-qdq')
                result=run_directory(root/f'out{version}',np.zeros((5,4),np.int8),quantized=True)
                self.assertEqual(len(result['output_int8']),5)
                # Restore embedded arrays before saving the next external model.
                model=onnx.load(path)

    def test_dynamic_quantization_never_falls_back_to_calibration(self):
        model,_=make_model()
        # A standard, valid dynamic op is still outside this hardware profile.
        model.graph.node.append(helper.make_node('DynamicQuantizeLinear',['input'],
                                               ['dynamic_q','dynamic_s','dynamic_z']))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            onnx.save(model,root/'dynamic.onnx')
            np.save(root/'calibration.npy',np.ones((4,4),np.float32))
            with self.assertRaises(ValueError):
                compile_model(root/'dynamic.onnx',root/'calibration.npy',root/'out')
            self.assertFalse((root/'out').exists())

@unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'optional ONNX Runtime not installed')
class ExportIntegrationTests(unittest.TestCase):
    def test_onnxruntime_symmetric_qdq_export(self):
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
        root=EXAMPLES
        class Reader(CalibrationDataReader):
            def __init__(self): self.rows=iter(np.load(root/'calibration.npy'))
            def get_next(self):
                row=next(self.rows,None)
                return None if row is None else {'input':row[None,:]}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            quantize_static(str(root/'model.onnx'),str(path/'ptq.onnx'),Reader(),
                            quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8,per_channel=False,
                            extra_options={'ActivationSymmetric':True,'WeightSymmetric':True})
            compiled=compile_model(path/'ptq.onnx',None,path/'compiled')
            report=json.loads((path/'compiled/import_report.json').read_text())
            self.assertIn('relu-identity-stage',[x['lowering'] for x in report['layers']])
            x=np.random.default_rng(99).uniform(-4,4,(33,4)).astype(np.float32)
            # Also verifies every per-k accumulator against independent integer IR.
            result=run_directory(path/'compiled',x)
            options=ort.SessionOptions()
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.intra_op_num_threads=1
            source=ort.InferenceSession(str(path/'ptq.onnx'),options,providers=['CPUExecutionProvider'])
            expected=source.run(None,{'input':x})[0]
            expected=np.rint(expected/compiled['output_scale']).astype(np.int32)
            actual=np.asarray(result['output_int8'],np.int32)
            # FLOAT32 Q/DQ and fixed-point target rounding can differ by 1 code.
            self.assertLessEqual(int(np.max(np.abs(actual-expected))),1)

class LayoutAndPipelineTests(unittest.TestCase):
    def test_default_outputs_and_debug_cleanup(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            model,_=make_model(qat=True)
            onnx.save(model,root/'model.onnx')
            manifest,out=compile_file(root/'model.onnx')
            self.assertEqual(set(p.name for p in out.iterdir()), {'program.bin','params.bin','sram.bin'})
            with self.assertRaisesRegex(ValueError,'--debug'):
                run_directory(out,np.ones((1,4)))
            compile_file(root/'model.onnx',debug=True)
            self.assertTrue((out/'program.asm').exists())
            self.assertTrue((out/'import_report.json').exists())
            (out/'notes.txt').write_text('user file')
            (out/'run.json').write_text('stale result')
            compile_file(root/'model.onnx')
            self.assertEqual(set(p.name for p in out.iterdir()),
                             {'program.bin','params.bin','sram.bin','notes.txt'})
            self.assertEqual((out/'notes.txt').read_text(),'user file')

    def test_actual_entry_point(self):
        import subprocess
        import sys
        path=Path(__file__).resolve().parent/'main.py'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            model,_=make_model()
            onnx.save(model,root/'model.onnx')
            result=subprocess.run([sys.executable,str(path),str(root/'model.onnx')],
                                  cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Compiled',result.stdout)
            self.assertEqual((root/'model_compiled/sram.bin').stat().st_size,SRAM_BYTES)

    def test_old_float_example_binary_golden(self):
        import hashlib
        image,program,*_=build(load_model(EXAMPLES/'model.onnx'),np.load(EXAMPLES/'calibration.npy'))
        # Normalize relocated byte addresses back to the old map, preserving
        # the pre-migration golden bytes rather than replacing their hashes.
        old_regions = {'weights': (0,0x2000), 'activations': (0x2000,0x3000),
                       'bias': (0x3000,0x3800), 'config': (0x3800,0x3900)}
        old_program = bytearray()
        for offset in range(0,len(program),16):
            ins = Instruction.decode(program[offset:offset+16])
            src, dest = ins.src, ins.dest
            if ins.op == Op.LOAD_OPERAND:
                region = ('activations','weights','bias','config')[dest & 3]
                src += old_regions[region][0] - REGIONS[region][0]
            elif ins.op == Op.STREAM_STORE:
                src += old_regions['activations'][0] - REGIONS['activations'][0]
            if ins.op == Op.STREAM_LOAD or (ins.op == Op.POST_PROC and ins.flags & 4):
                dest += old_regions['activations'][0] - REGIONS['activations'][0]
            # Encode fields directly because old addresses are intentionally
            # invalid under the new map's instruction validators.
            word = (int(ins.op)<<124)|(src<<92)|(dest<<60)|(ins.dims<<32)|ins.flags
            old_program.extend(word.to_bytes(16,'little'))
        old_image = bytearray(0x4000)
        for region,(start,end) in old_regions.items():
            base = REGIONS[region][0]
            old_image[start:end] = image[base:base+end-start]
        old_image[0x3900:0x3900+len(old_program)] = old_program
        self.assertEqual(hashlib.sha256(old_program).hexdigest(), 'e9f7dcb1de4244089919fdced460eee306d604ea2fa047e388bdf3d7726782cd')
        self.assertEqual(hashlib.sha256(old_image).hexdigest(), 'dd7c96a0a5cc88b1f8fd4b466095cce8c9ee5b4c9529043a145125accdba08e0')


if __name__ == '__main__':
    unittest.main()
