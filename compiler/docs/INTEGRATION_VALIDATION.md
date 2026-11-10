# Directory integration

This release contains only `compiler/`, for placement under `Tiny-NPU/`.
The executable is `compiler/bin/tiny-npu-compiler`, not a sibling root `bin/`.
All Makefile targets work from inside `compiler/`; no root Makefile is required.
Source execution remains `python3 compiler/npucc.py ...` from the project root.

No C-model, training files, root build files or architecture diagrams are in
this release. ISA v1 is unchanged. C-model compatibility has not been established
by these compiler tests; that model must implement `docs/ISA.md` before running
the generated image. See `VALIDATION.md` for compiler-only checks.
