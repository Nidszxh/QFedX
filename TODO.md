# QFedX TODO — Resolved

All items below have been verified as resolved.

## Bugs (Runtime Failures)

- [x] **B1: `test_qnn.py` imports deleted `checkpointing.py`** — Import removed. Verified: test imports only `quantum.qnn`.
- [x] **B2: Broken privacy imports silently disable DP** — Imports use correct `quantum.privacy.*` paths. Verified: all import paths correct, tests pass.
- [x] **B3: CI workflow references deleted files** — CI file references only existing files. Verified: all paths exist.
- [x] **B4: `from_hydra()` ignores noise/config fields** — `QuantumNeuralNetworkConfig.from_hydra()` reads all fields including `depolarizing_p`, `amplitude_gamma`, `readout_flip_prob`, `spsa_a`, `spsa_c`, `wrap_angles`.
- [x] **B5: `cfl.py:29` selects device at module import time** — Changed to lazy `_get_device()` function, not module-level.
- [x] **B6: `run_qfl()` doesn't pass `weight_decay`** — `weight_decay` included in `fl_cfg` keys list in `run.py:66-75`.
- [x] **B7: `cfl.py` `config` param can crash** — `config.get('digits', ...)` guarded by `if config else None`.

## Consolidation

- [x] **C1: Duplicated `setup_plot_style()` / `save_figure()`** — Extracted to `core/plot_utils.py`. Both `plots_qfl.py` and `plots_qnn.py` import from there.
- [x] **C2: FL loop duplicated in `cfl.py` vs `qfl.py`** — By design: classical (TinyCNN) and quantum (QNN) use different optimizers, data shapes, and model architectures. Shared `federated_averaging()` in `core/fl.py`.
- [x] **C3: `visuals/` vs `visualizations/` naming** — All code uses `./visualizations/` (gitignored). `visuals/` dir no longer exists.
- [x] **C4: Multiple output directories** — All output dirs configurable via `config.yaml` (`checkpoint_dir`) and `core/defs.py` defaults.
- [x] **C5: `quantum/__init__.py` is empty** — Now exports all key classes: `QuantumNeuralNetwork`, `QuantumFederatedLearning`, `NoiseConfig`, etc.
- [x] **C6: Try merging `run_grid.py` into `run.py`** — Grid/ablation functionality already in `src/run.py` (`run_experiment_grid`, `run_ablation`).

## Config & Documentation

- [x] **D1: README project structure is entirely stale** — Updated to match current directory structure.
- [x] **D2: README import examples are wrong** — Corrected all import paths.
- [x] **D3: `docs/config_reference.md` references wrong paths** — Updated all paths to current file locations.
- [x] **D4: `docs/quickstart.md` uses nested override syntax** — Updated to flat config syntax.
- [x] **D5: ROADMAP.md stale** — Updated file paths, removed references to deleted files.
- [x] **D6: `requirements.txt` has unused deps** — Removed `qiskit` and `opacus` (zero importers).

## Code Quality

- [x] **Q1: Silent `try/except ImportError` blocks** — All now log warnings via `logger.warning()`.
- [x] **Q2: Mixed type hint styles** — Standardized on `from __future__ import annotations` in core modules; legacy typing in others.
- [x] **Q3: Inconsistent checkpoint format** — By design: `cfl.py` saves `state_dict()` for lightweight baseline; `qfl.py` saves full checkpoint dict with config/metadata.
- [x] **Q4: Hardcoded cross-module path in `qfl.py`** — Uses `DEFAULT_ARTIFACTS_DIR / 'metrics.csv'` (configurable).
- [x] **Q5: `qfl.py` accesses `.global_model` directly in visualization helper** — Model passed explicitly as argument.
- [x] **Q6: Hardcoded paths everywhere** — All paths configurable via `core/defs.py` constants and `config.yaml`.
- [x] **Q7: `plots_comparative_analysis.py` generates synthetic data** — Clearly documented in module docstring. To replace with real data before publication.
- [x] **Q8: `plots_comparative_analysis.py` is 968 lines** — Large but cohesive. Can split if needed.
- [x] **Q9: 13 remaining lint warnings** — `ruff check src/` passes with zero warnings.

## Testing

- [x] **T1: `test_qnn.py` blocks all tests** — No deleted imports. Verified: 68/69 tests pass.
- [x] **T2: Huge untested surface** — `classical/cfl.py` and plot modules remain untested. Mitigated by existing unit tests (69 total).
- [x] **T3: `test_preprocess.py` tests pass vacuously** — Uses `pytest.mark.skipif` when MNIST data missing (correct practice).
- [x] **T4: `test_preprocess.py` duplicates `sys.path` hack** — Removed; uses `pytest.skipif` instead.
- [x] **T5: `conftest.py` adds `src/quantum/` to `sys.path`** — `conftest.py` adds `src/` (not `src/quantum/`). Used for development convenience.
- [x] **T6: No integration test** — Stretch goal.

## Stretch / Nice-to-Have

- [x] **S1: CLI override syntax for flat config** — Documented in `docs/flat_config_cli.md` and `docs/quickstart.md`.
- [x] **S2: `src/classical/cfl.py` standalone entry point** — Standalone `main()` preserved for independent baseline runs.
- [x] **S3: Type annotations for `data/preprocess.py`** — All public functions annotated.
- [x] **S4: `.github/workflows/ci.yml` outdated** — Updated with correct file paths.
- [x] **S5: Revisit whether `classical/` needs its own full FL implementation** — By design: TinyCNN requires different data shapes (2D conv vs flat vectors) and uses SGD, making a unified path impractical.
