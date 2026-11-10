"""Build a single executable on the target OS. Run in a dedicated build venv.

    cd compiler
    python -m pip install -r requirements-build.txt
    python build_binary.py

The compiler stays Python source; PyInstaller bundles its interpreter and libs.
Linux builds are not Windows executables. Build on the oldest supported Linux.
"""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    compiler = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
                    "--distpath", str(compiler / "bin"),
                    "--workpath", str(compiler / "build" / "compiler-binary"),
                    str(compiler / "tiny-npu-compiler.spec")], check=True)
