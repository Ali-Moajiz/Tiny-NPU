"""Optional packaging tool. This does NOT compile a neural network.

It bundles main.py and its dependencies into bin/tiny-npu-compiler.
Run with the same Python environment used for development; see README.md.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys


def main():
    if importlib.util.find_spec('PyInstaller') is None:
        raise SystemExit('Install build tools: python -m pip install pyinstaller==6.16.0 pyinstaller-hooks-contrib==2025.9')
    root = Path(__file__).resolve().parent
    build = root / 'build'
    build.mkdir(exist_ok=True)
    # Keep the generated PyInstaller recipe with temporary build products.
    spec = build / 'compiler.spec'
    spec.write_text('''from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
root = Path(SPECPATH).parent
a = Analysis([str(root / 'main.py')], pathex=[str(root)],
             binaries=[], datas=collect_data_files('onnx'),
             hiddenimports=['onnx.onnx_cpp2py_export'],
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['tkinter', 'matplotlib', 'scipy', 'torch', 'tensorflow',
                       'onnxruntime', 'onnx.backend.test', 'IPython', 'test_compiler'],
             noarchive=False)
# Use the target system's C++ runtime instead of bundling the build host's copy.
# Still build Python/NumPy/ONNX against the oldest supported glibc; see README.
a.binaries = [x for x in a.binaries if Path(x[0]).name not in ('libstdc++.so.6', 'libgcc_s.so.1')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='tiny-npu-compiler',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
''')
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm',
                    '--distpath', str(root / 'bin'), '--workpath', str(build / 'work'),
                    str(spec)], check=True)


if __name__ == '__main__':
    main()
