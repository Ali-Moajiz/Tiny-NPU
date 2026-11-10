"""Test the executable-style CLI without relying on the build system."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from tiny_npu.cli import main


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


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
