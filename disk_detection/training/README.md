# Original supervised disk-detector training implementation

This directory preserves the original r4 training implementation and its two
native extensions byte-for-byte. It is included for method inspection and full
retraining when the supervised training records are available. The packaged
checkpoint and the tested inference/label-generation entry points remain one
directory above.

## Contents

- `disk_detector_train.py`: 150-epoch mixed experimental/simulation training.
- `disk_detector_dataset.py`: loader for supervised legacy PKL records.
- `disk_detector_model.py`: original ConvNeXt V2 deformable set predictor.
- `diffraction_transform.py`: experimental and simulation augmentations.
- `disk_detector_inference.py`: original development-time inference script.
- `ops/`: multi-scale deformable-attention CUDA/C++ source and the original
  CPython 3.12 Windows binary.
- `vendor/torch-linear-assignment/`: the original locally supplied matching
  extension. The active matcher in `disk_detector_train.py` uses SciPy, but the
  module is still imported by that unchanged script.

## Environment

From the top-level `baselines/` directory, install the common dependencies and
the two native extensions:

```text
python -m pip install -r requirements-disk-training.txt
python -m pip install ./disk_detection/training/vendor/torch-linear-assignment
python -m pip install ./disk_detection/training/ops
```

Building the extensions requires a CUDA-enabled PyTorch installation, a CUDA
toolkit compatible with that installation, and a supported C++ compiler. The
included `MultiScaleDeformableAttention.cp312-win_amd64.pyd` is the original
Windows CPython 3.12 binary; rebuilding is safer when Python, PyTorch, or CUDA
versions differ.

## Required supervised data

The unchanged training script expects these paths relative to its working
directory:

```text
dataset/sim_dataset_test/
dataset/exp_dataset_v1/
```

Each PKL must contain `item_list`. Every item must contain:

- `dp`: one two-dimensional diffraction pattern;
- `qi_list`: padded diffraction-row disk coordinates;
- `qj_list`: padded diffraction-column disk coordinates;
- `-1` in `qi_list` and `qj_list` for unused positions.

These supervised source records are distinct from the model-predicted labels
released for 4D-ImageNet and are not embedded in this compact package. They must
be supplied before running `disk_detector_train.py`. This distinction prevents
the r4 model from being presented as though it could be independently retrained
from its own pseudo-label outputs.

