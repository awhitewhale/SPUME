# SPUME

Official implementation of **SPUME: Surface-Persistence Modeling for
Transient-Resistant Underwater Multi-View Reconstruction**.

[[Project page](https://awhitewhale.github.io/SPUME/)]

SPUME uses provisional cameras and geometry from a frozen feed-forward backbone
to compare point, depth, feature, normal, and multi-view support evidence at
predicted surface anchors. Robust window-wise normalization and weighted fusion
produce a Surface-Support Deficit Field (SSDF), which conditions an
identity-initialized, bounded log-depth refiner.

## Repository layout

```text
configs/       experiment configurations
datasets/      FLSea RGB/depth pairing and deterministic windows
diagnostics/   cross-view evidence and SSDF construction
models/        SSDF-conditioned bounded depth adapter
scripts/       frozen-backbone cache construction
training/      experiment-specific training programs
evaluation/    result summarization
tests/         unit tests
docs/          static project page and web-ready assets
tools/         project-page asset preparation
train.py       cache-based adapter training entry point
test.py        checkpoint evaluation entry point
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Cache construction requires a working
[Wat3R](https://github.com/Junyi42/Wat3R) installation, its checkpoint, and the
FLSea data. Training and evaluation from an existing cache only require PyTorch,
NumPy, and PyYAML.

## Quick verification

The built-in deterministic example checks the complete adapter training and
evaluation path without downloading a dataset:

```bash
python train.py --demo --epochs 2 --output-dir outputs/demo
python test.py --demo --checkpoint outputs/demo/checkpoint.pt
pytest -q
```

The commands run on CUDA when available and otherwise use the CPU.

## Data preparation

FLSea sequences are expected in the following layout:

```text
FLSea_ROOT/
  flatiron/
    imgs/*.tiff
    depth/*_SeaErra_abs_depth.tif
  horse_canyon/
  tiny_canyon/
  u_canyon/
```

Edit a configuration under `configs/` so that `upstream_root`, `checkpoint`,
`flsea_root`, `cache_dir`, and `output_dir` point to local paths. Build the
frozen-backbone cache with:

```bash
python scripts/build_adapter_cache.py --config configs/adapter_v0.yaml
```

Each cache file is an `.npz` archive containing `rgb`, `depth`, `confidence`,
`risk`, and `gt`. Arrays use a leading frame dimension. The cache decouples
adapter optimization from the expensive frozen-backbone forward pass.

## Training

Train the SSDF-conditioned depth adapter on cache files whose names begin with
`train_`:

```bash
python train.py \
  --cache-dir /path/to/cache \
  --output-dir outputs/spume \
  --epochs 5 \
  --batch-size 4 \
  --device cuda
```

The output directory contains `checkpoint.pt`, `training_metrics.csv`, and
`training_summary.json`. The adapter starts as an exact identity mapping and
learns a bounded log-depth residual while the backbone outputs remain fixed.

The original matched-control experiment is also retained:

```bash
python training/train_output_adapters.py --config configs/adapter_v0.yaml
```

## Evaluation

Evaluate a checkpoint on held-out cache files:

```bash
python test.py \
  --checkpoint outputs/spume/checkpoint.pt \
  --cache-dir /path/to/test-cache \
  --split test \
  --output outputs/spume/test_metrics.json \
  --device cuda
```

If no files match `<split>_*.npz`, `test.py` reports the expected pattern and
exits. Reported metrics are raw AbsRel, least-squares scale-aligned AbsRel,
RMSE, delta-1 accuracy, and valid coverage.

## Reproducibility notes

- RGB/depth pairs are matched by exact filename stem rather than independent
  sorting.
- Random seeds are applied to Python, NumPy, and PyTorch.
- Invalid, non-finite, and out-of-range depths are excluded consistently.
- Generated caches, checkpoints, and output directories are ignored by Git.
- Dataset licenses and access conditions remain those of the original providers.
