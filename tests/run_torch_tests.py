"""Run every PyTorch code block published on the site as a standalone test.

The site was written in an environment without PyTorch, so these blocks were
verified only through equivalent NumPy implementations. Run this once on any
machine with PyTorch (CPU is enough):

    pip install torch numpy
    python tests/run_torch_tests.py            # all modules
    python tests/run_torch_tests.py m04 m14    # selected modules

Each file is self-contained, uses asserts, and exits 0 on success.
Tests that need CUDA or a newer PyTorch print SKIP and exit 0.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent / "torch"


def main() -> int:
    want = set(sys.argv[1:])
    files = sorted(p for p in HERE.glob("m*/*.py") if not want or p.parent.name in want)
    if not files:
        print("no test files found")
        return 1
    failed = []
    for f in files:
        t0 = time.time()
        r = subprocess.run([sys.executable, f.name], cwd=f.parent, capture_output=True, text=True, timeout=900)
        dt = time.time() - t0
        status = "PASS" if r.returncode == 0 else "FAIL"
        if "SKIP" in r.stdout and r.returncode == 0:
            status = "SKIP"
        print(f"{status}  {f.parent.name}/{f.name}  ({dt:.1f}s)")
        if r.returncode != 0:
            failed.append(f)
            tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
            for line in tail:
                print("      " + line)
    print(f"\n{len(files) - len(failed)}/{len(files)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
