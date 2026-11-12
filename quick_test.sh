#!/usr/bin/env bash
# Run from the project root: ./quick_test.sh
# Prerequisites: C compiler, Make, Python 3.10-3.12 with venv support, and
# network access for first-time dependency installation from requirements.txt.
# Optional: PYTHON=/path/to/python3 LOG=flow ./quick_test.sh
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_root"

main() {
    python_bin="$(command -v "${PYTHON:-python3}")" || {
        printf 'Python not found. Set PYTHON to your Python executable.\n' >&2
        exit 1
    }
    if [[ "$python_bin" != /* ]]; then
        python_bin="$project_root/$python_bin"
    fi
    command -v make >/dev/null || { printf 'Make is required.\n' >&2; exit 1; }

    log_mode="${LOG:-none}"
    case "$log_mode" in
        none|flow|ports|both) ;;
        *) printf 'LOG must be none, flow, ports, or both.\n' >&2; exit 1 ;;
    esac

    trap 'printf "Quick test failed at line %s.\n" "$LINENO" >&2' ERR

    printf '\n[1/6] Set up the Python environment and install requirements\n'
    # Use a selected virtual environment, or create/reuse the project's .venv.
    if ! "$python_bin" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)'; then
        if ! "$python_bin" -m venv "$project_root/.venv"; then
            printf 'Virtual environment setup failed. Install Python venv support (python3-venv on Ubuntu/Debian), then rerun.\n' >&2
            exit 1
        fi
        python_bin="$project_root/.venv/bin/python3"
    fi
    "$python_bin" -m pip install -r "$project_root/requirements.txt"

    printf '\n[2/6] Train and export the 4 -> 100 -> 100 -> 1 model\n'
    "$python_bin" test-dnn-large/train.py

    printf '\n[3/6] Rebuild the compiler executable\n'
    make -C compiler binary PYTHON="$python_bin"

    printf '\n[4/6] Compile the trained ONNX model (batch 4)\n'
    ./compiler/bin/tiny-npu-compiler test-dnn-large/artifacts/model.onnx \
        --calibration test-dnn-large/artifacts/model.calibration.npy \
        --out output-large --batch 4 --debug

    printf '\n[5/6] Build the C model with TB=1\n'
    make -C c-model all TB=1 BATCH=4 POST_LATENCY=1 SANITIZE=0 LOG="$log_mode"

    printf '\n[6/6] Run 100 random batches with seed 42\n'
    make -C c-model run TB=1 BATCH=4 POST_LATENCY=1 SANITIZE=0 LOG="$log_mode" \
        SRAM=../output-large/sram.hex \
        ARGS="--runs 100 --seed 42 --input-features 4 --output-features 1"

    printf '\nQuick test completed. Full log: %s/quick_test.log\n' "$project_root"
    # This smoke run checks successful execution; it does not compare predictions
    # with a numerical reference or measure model accuracy.
}

# Capture stdout and stderr for every stage, including dependency errors.
# pipefail preserves a failed stage even when tee succeeds.
main 2>&1 | tee "$project_root/quick_test.log"
