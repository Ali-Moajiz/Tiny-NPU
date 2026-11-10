"""Optional test of a real PTQ exporter, including its separate ReLU boundary.

Install requirements-test.txt to enable. No torch/TF training is claimed here.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tiny_npu.compiler import compile_model
from tiny_npu.cli import run_directory


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'optional ONNX Runtime not installed')
class ExportIntegrationTests(unittest.TestCase):
    def test_onnxruntime_symmetric_qdq_export(self):
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
        root=Path(__file__).resolve().parents[1]
        class Reader(CalibrationDataReader):
            def __init__(self): self.rows=iter(np.load(root/'examples/calibration.npy'))
            def get_next(self):
                row=next(self.rows,None)
                return None if row is None else {'input':row[None,:]}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            quantize_static(str(root/'examples/model.onnx'),str(path/'ptq.onnx'),Reader(),
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


if __name__=='__main__':unittest.main()
