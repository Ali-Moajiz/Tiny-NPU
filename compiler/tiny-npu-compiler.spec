# PyInstaller executable build recipe. SPECPATH is this compiler/ directory.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH)
a = Analysis([str(root / 'npucc.py')], pathex=[str(root)],
             binaries=[], datas=collect_data_files('onnx'),
             hiddenimports=['onnx.onnx_cpp2py_export'],
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['tkinter', 'matplotlib', 'scipy', 'torch', 'tensorflow', 'onnxruntime',
                       'onnx.backend.test', 'IPython'], noarchive=False)
# The target distro supplies libstdc++. Bundling Ubuntu 24's copy would impose
# its newer glibc on Ubuntu 22.04 even when ONNX only needs older C++ symbols.
# Check all collected ELF version requirements before claiming compatibility.
a.binaries = [x for x in a.binaries if Path(x[0]).name not in ('libstdc++.so.6', 'libgcc_s.so.1')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
          name='tiny-npu-compiler', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=True)
