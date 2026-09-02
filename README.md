# 4D-ImageNet reproducible baselines

This folder is a self-contained code package for the baseline methods and data
construction utilities described with 4D-ImageNet. Open this folder itself as a
VS Code workspace. The included examples use real experimental and simulated
diffraction patterns copied or extracted from the project data; no diffraction
feature was generated for the examples.

The original implementations were copied byte-for-byte into this package. The
only new Python entry points are package-relative example and verification
wrappers. Running an example writes only to `outputs/`.

## Contents

```text
baselines/
|-- .vscode/launch.json
|-- README.md
|-- requirements.txt
|-- requirements-data-pipeline.txt
|-- requirements-disk-training.txt
|-- verify_package.py
|-- source_integrity.json
|-- package_manifest.json
|-- sample_data/
|   |-- manifest.json
|   |-- experimental/
|   |-- simulation/
|   `-- mixed_train/
|-- data_preparation/
|   |-- transform_format.py
|   |-- extract_pkl.py
|   |-- merge_pkl.py
|   |-- simulation_with_orientation.py
|   |-- audit_4d_imagenet.py
|   |-- visualize_pkl.py
|   |-- visualize_merged_pkl.py
|   `-- cif/
|-- disk_detection/
|   |-- model.py
|   |-- configs.py
|   |-- infer_4d_imagenet.py
|   |-- generate_exp_labels.py
|   |-- r4_weights.pth
|   |-- model_config.json
|   |-- inference_example.json
|   `-- training/
|       |-- disk_detector_train.py
|       |-- disk_detector_dataset.py
|       |-- disk_detector_model.py
|       |-- diffraction_transform.py
|       |-- ops/
|       `-- vendor/torch-linear-assignment/
`-- masked_reconstruction/
    |-- dataset.py
    |-- masks.py
    |-- metrics.py
    |-- models.py
    |-- masked_reconstruction.py
    |-- mixed_masked_reconstruction.py
    |-- run_inference_example.py
    |-- mixed_domain_epoch48.pth
    |-- model_config.json
    |-- split_manifest.json
    `-- test_metrics.json
```

## Installation

Python 3.12 was used for package verification. Install the two neural-network
baselines with:

```text
python -m pip install -r requirements.txt
```

The format-conversion and simulation tools require additional packages:

```text
python -m pip install -r requirements-data-pipeline.txt
```

The original supervised disk-detector training implementation additionally
requires native extensions. Its separate installation and input contract are
documented in `disk_detection/training/README.md`.

The data-pipeline file pins `zarr==2.15.0`; newer zarr 2.x combinations were
observed to fail during abTEM import in one existing environment.

PyTorch CUDA wheels depend on the local CUDA driver. Install the appropriate
PyTorch build first when the default package index does not provide the required
accelerator build. Both packaged baseline examples also run on CPU.

## One-click VS Code runs

Open `baselines/` in VS Code, select **Run and Debug**, and choose:

- **Verify Standalone Baseline Package**: validates JSON, sample files, hashes,
  imports, and strict loading of both checkpoints.
- **Disk Detection - Packaged Experimental Example**: predicts the direct beam
  and Bragg-disk centres for the included real Pd pattern.
- **Masked Reconstruction - Packaged Experimental and Simulation Examples**:
  loads the epoch-48 checkpoint and evaluates one real experimental and one
  deposited simulation pattern using the published spot-mask transform.
- **Mixed-domain Training - Tiny Smoke Test**: runs one training epoch on three
  real experimental acquisitions and three deposited simulation sources. This
  verifies the complete mixed-domain indexing, grouped splitting, training,
  validation, checkpointing, and test path; it is not a performance benchmark.
- **Transform Format - GUI**: opens the ARINA H5 / Merlin MIB conversion GUI.
- **Multislice Simulation - One Ag Pattern**: runs the unchanged abTEM simulator
  through a package-relative wrapper. This optional job requires a working
  abTEM GPU environment and is substantially heavier than the baseline examples.

## Reproducing the reported results

The included tiny data are smoke-test inputs only. The reported r4 predictions
are reproduced from `disk_detection/r4_weights.pth`; the complete dataset can be
processed by pointing `infer_4d_imagenet.py` or `generate_exp_labels.py` to the
released PKL files. The epoch-48 mixed-domain result is represented by
`masked_reconstruction/mixed_domain_epoch48.pth`, its exact acquisition/file
split in `split_manifest.json`, and held-out metrics in `test_metrics.json`.

Full masked-reconstruction retraining requires the complete experimental and
simulation data described by those manifests. Full r4 detector retraining also
requires its independently supervised legacy `item_list` PKLs; these are not the
model-predicted 4D-ImageNet labels and are not embedded here. Raw ARINA or Merlin
files are needed only when repeating format conversion and 5,000-pattern
curation.

## Scope

This package contains executable implementations for format conversion,
experimental pattern curation, PKL merging, multislice simulation, dataset audit,
r4 disk annotation, original r4 training source, and mixed-domain masked
reconstruction. It does not embed the full 174,000-pattern data record, the
145,000-pattern model-predicted label collection, raw detector files, or the
independent supervised records used to train r4; those large records should be
distributed through the associated data repository. The packaged examples make
the released checkpoints and downstream workflows testable without those large
downloads.

The package has been exercised from `baselines/` as an independent working
directory: checkpoint integrity verification, r4 inference on a real Pd pattern,
masked reconstruction on real experimental and deposited simulation patterns,
and a one-epoch mixed-domain training/validation/test smoke run all complete.
Multislice generation and original r4 retraining remain environment/data-gated
heavy workflows rather than compact smoke tests.

`package_manifest.json` provides a machine-readable mapping from each article
workflow to its entry point, included assets, verification status, and any
external data or hardware requirement.
