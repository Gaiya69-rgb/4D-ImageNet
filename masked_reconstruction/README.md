# Mixed-domain masked-reconstruction baseline

This directory packages the epoch-48 Small U-Net checkpoint used for the
mixed experimental-simulation masked-reconstruction benchmark. The model was
trained on experimental and simulated 4D-STEM diffraction patterns and selected
by the lowest acquisition-level validation loss. The complete model, data
loading, masking, training, metric, checkpoint-loading and example-inference
implementation is included in this directory.

## Files

- `mixed_domain_epoch48.pth`: byte-identical copy of
  `encoder/outputs/mixed_exp_sim_v1/model_best_mixed.pt`.
- `model_config.json`: model, preprocessing, mask, optimization, and checkpoint
  metadata.
- `split_manifest.json`: every acquisition or simulation file assigned to train,
  validation, experimental test, or simulation test.
- `test_metrics.json`: metrics from the independent experimental and simulation
  test sets.
- `dataset.py`, `masks.py`, `metrics.py`, `models.py`,
  `masked_reconstruction.py`, and `mixed_masked_reconstruction.py`:
  byte-identical copies of the maintained implementation.
- `run_inference_example.py`: package-relative evaluation of the included real
  experimental and deposited simulation patterns.

## Method

Each diffraction pattern is clipped to nonnegative intensity, transformed with
`log1p`, normalized by its own maximum, and resized to 192 x 192 when needed.
The model receives two channels, `[x * (1 - M), M]`, where `M` is the binary
mask. One to four circular masks are centred on detected intensity peaks while a
central exclusion radius prevents masking the direct beam.

The Small U-Net contains two encoder stages with 16 and 32 channels, a 64-channel
bottleneck, and two symmetric decoder stages with skip concatenation. Every
convolution block contains two 3 x 3 convolution, batch-normalization, and ReLU
sequences. A 1 x 1 convolution and sigmoid produce the one-channel reconstruction.

Training minimizes intensity-weighted MSE only inside the masked region. The
weight per masked pixel is `1 + 8 * target`, so bright diffraction features are
not overwhelmed by the dark background.

## VS Code use

The packaged implementation is in:

- `masked_reconstruction/models.py`
- `masked_reconstruction/dataset.py`
- `masked_reconstruction/masks.py`
- `masked_reconstruction/metrics.py`
- `masked_reconstruction/mixed_masked_reconstruction.py`

Open the top-level `baselines/` folder in VS Code. **Masked Reconstruction -
Packaged Experimental and Simulation Examples** loads the frozen checkpoint and
writes an example figure and JSON metrics under `outputs/`. **Mixed-domain
Training - Tiny Smoke Test** exercises the complete grouped train/validation/test
path on the included compact real-data subset. The training script probes GPU
memory, halves the batch size after CUDA OOM, increases gradient accumulation,
and creates a versioned output directory instead of overwriting the published
checkpoint.

For evaluation-only loading in a Python file opened from the `baselines/` root:

```python
from pathlib import Path
import sys
import torch

baseline_dir = Path("masked_reconstruction").resolve()
sys.path.insert(0, str(baseline_dir))
from models import SmallUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "masked_reconstruction/mixed_domain_epoch48.pth",
    map_location=device,
    weights_only=False,
)
model = SmallUNet(in_channels=2, out_channels=1, base_channels=16).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

Use `DiffractionPatternDataset.preprocess`, `make_mask`, and `prepare_input` from
the packaged implementation to reproduce the published input transform exactly.

## Split and metric interpretation

Splitting is performed by experimental acquisition or simulation source file,
not by individual pattern. This prevents patterns from one merged acquisition
from appearing in more than one split. The independent experimental test set
contains six held-out acquisitions covering Ag, Au, Au-Ag, CoO, Pd, and ZnO. The
simulation test set contains held-out CeO2 and SrTiO3 files.

Masked MSE, masked MAE, and intensity-weighted masked MSE are evaluated only over
masked pixels. PSNR is computed from masked MSE with a normalized maximum of 1.
SSIM was disabled in the recorded run and is therefore `null` in
`test_metrics.json` rather than a measured value.
