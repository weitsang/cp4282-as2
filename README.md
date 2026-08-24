# CP4282 Gaussian Splatting — Assignment 2 (Training)

Starter code and data for the CP4282 Gaussian Splatting **training** assignment.

This repository intentionally contains incomplete teaching implementations. The missing sections
are the work: read the corresponding unit in the course notes, implement the marked functions,
and use the supplied checks before moving on.

## Setup

```bash
git clone https://github.com/weitsang/cp4282-as2.git
cd cp4282-as2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository contains a small Lego dataset under `data/lego/`:

- `train/` contains posed training images and `transforms_train.json`.
- `test/` contains held-out images and `transforms_test.json`.
- `init.ply` is a small starting point for the full-training assignment.

## The assignment

1. `3dgs_trainer.py`: implement the backward rendering pass. Two kernels are marked TODO,
   `render_backward` for the dense path and `render_sparse_backward` for the sampled one.

```bash
python 3dgs_trainer.py config/3dgs_training_gpu.yaml
```

Check your gradients against finite differences before judging a training run:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

As shipped this check fails, reporting an analytic gradient of zero: nothing accumulates into the
gradient buffers until you write the kernels. Training also runs to completion without improving,
for the same reason.

The annotated walkthrough explains the whole file without giving the TODO solution:

- `3dgs_trainer_annotated.md`

Shared and support files:

- `shared/trainable_gaussian.py`: trainable splat parameter storage used by the trainer
  (documented in `shared_annotated.md`)
- `shared/training_config.py`: the YAML schema, defaults and validation
- `gaussian_first_tile_workspace_gpu.py`: tile record construction used by the trainer
- `3dgs_renderer_v1.py`: the reference CPU renderer the synthetic trainers reuse
- `config/`: starter YAML files for training runs
- `3dgs_1_syn_trainer.py`: small Unit 8 trainer for one synthetic Gaussian
- `3dgs_k_syn_trainer.py`: small Unit 8 trainer for several synthetic Gaussians

## Evaluating your output

`3dgs_evaluator.py` renders every camera in a manifest, saves the images, and reports the metrics
you ask for:

```bash
python 3dgs_evaluator.py outputs/warp-training.ply data/lego
```

It decides three things for you rather than making you pick a script:

- **Device.** `--arch auto` (the default) uses CUDA when Warp reports it and CPU otherwise. Force
  it with `--arch cpu` or `--arch gpu`; asking for gpu without CUDA is an error, not a slow
  fallback.
- **Appearance.** If a `<ply-stem>.sh.npz` sidecar sits beside the PLY, the render uses degree-2
  view-dependent colour. `--sh-degree 0` ignores it, `--sh-degree 2` requires it, and `--sh-npz`
  points at one stored elsewhere. This assignment produces no sidecar, so this stays off.
- **Metrics.** `--metrics` takes any of `psnr`, `ssim`, `lpips`; all three run by default. LPIPS
  needs PyTorch, so drop it with `--metrics psnr ssim` if you have not installed the optional
  requirements.

Use `--manifest transforms_train.json` to score against the training split instead of the
held-out test split, and `--width`/`--height` to render at a different resolution than the
default.

## Running checks

```bash
python -m compileall scripts shared 3dgs_trainer.py 3dgs_gradient_check_gpu.py gaussian_first_tile_workspace_gpu.py 3dgs_1_syn_trainer.py 3dgs_k_syn_trainer.py image_metrics.py 3dgs_evaluator.py 3dgs_renderer_v1.py
python scripts/check_setup.py
```

After implementing `3dgs_trainer.py`'s backward pass, verify it with a finite-difference gradient
check:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
python 3dgs_gradient_check_gpu.py --device cuda:0
```

Use `--help` on each script for its command-line arguments. Start with low resolution and a small
iteration count while debugging.
