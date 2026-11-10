"""Imported Q/DQ semantics checked against ONNX reference execution."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
from onnx.reference import ReferenceEvaluator
from tiny_npu.qdq import Importer
from tiny_npu.compiler import compile_model, build_quantized
from tiny_npu.cli import main, run_directory
from tiny_npu.interpreter import execute, pack_tensor, unpack_tensor
from tiny_npu.quantize import int8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from make_quantized_examples import make_model


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
                self.assertEqual(main([str(root/'net.onnx')]),0)
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


if __name__=='__main__':unittest.main()
