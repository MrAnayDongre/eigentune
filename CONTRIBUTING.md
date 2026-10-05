# Contributing

EigenTune is small on purpose. The most useful contributions are a new backend, a measurement, or a test that
breaks something.

## Setup

```bash
pip install torch                      # or the CPU wheel
pip install -e ".[dev]"                # add ,triton for the Triton backend
pytest -q                              # CPU suite; GPU and ROCm tests skip themselves
ruff check . && ruff format --check .
```

## Ground rules

* **The reference is the oracle.** `kernels/reference.py` is plain PyTorch. Any other backend must agree with it (and an
  fp64 oracle) in `tests/kernels`, in fp32, fp16 and bf16, on rectangular, odd and tail shapes.
* **A claim needs a measurement.** Do not add a speed, memory or quality claim without a script in `benchmarks/` and its
  JSON in `benchmarks/results/`. `benchmarks/report.py` regenerates `docs/benchmarks.md` from those files, so the numbers cannot
  drift from the results.
* **Say what was not tested.** `docs/compatibility.md` lists what has been run; extend it honestly, including the
  failures.
* **A backend that loses must not be selected.** Dispatch thresholds live in `kernels/policy.py` and each one points at a benchmark.

## Adding a backend

See [docs/kernels.md](docs/kernels.md). In short: implement the five methods, `register_backend(...)`, run
`pytest tests/kernels`, then `python benchmarks/kernels.py` to see where it wins.

## AMD / ROCm

Nothing has been run on AMD hardware ([docs/rocm.md](docs/rocm.md)). If you have a ROCm GPU, running `pytest tests/kernels` and
`python benchmarks/kernels.py` and opening a PR with the output is the most valuable thing you can do.

## Benchmarks on a laptop

Long GPU runs on a laptop can overheat the machine. `benchmarks/_thermal.py` pauses work when the CPU or GPU gets hot; keep
it enabled, and run one GPU process at a time.
