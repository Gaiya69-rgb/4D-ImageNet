from __future__ import annotations

import ast
import hashlib
import json
import sys
import tokenize
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
EXPECTED_HASHES = {
    "disk_detection/r4_weights.pth": "f4e54e53528170e49cb9e60ec311e9fa7dc71c54436cf9353a6af2d17337adbd",
    "masked_reconstruction/mixed_domain_epoch48.pth": "54def410be0419d58ce7913eafe112bc0bf2ffc6ace8b80e76f120340cc10018",
}
JSON_FILES = [
    "disk_detection/model_config.json",
    "disk_detection/inference_example.json",
    "masked_reconstruction/model_config.json",
    "masked_reconstruction/split_manifest.json",
    "masked_reconstruction/test_metrics.json",
    "sample_data/manifest.json",
    "source_integrity.json",
    "package_manifest.json",
    "disk_detection/full_label_manifest.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            failures.append(f"hash mismatch: {relative}")
        print(f"[hash] {relative}: {actual}")

    for relative in JSON_FILES:
        with (ROOT / relative).open("r", encoding="utf-8") as handle:
            json.load(handle)
        print(f"[json] {relative}: valid")

    source_integrity = json.loads((ROOT / "source_integrity.json").read_text(encoding="utf-8"))
    for source_file in source_integrity["files"]:
        path = ROOT / source_file["packaged_file"]
        if sha256(path) != source_file["sha256"]:
            failures.append(f"source hash mismatch: {source_file['packaged_file']}")
        if path.suffix == ".py":
            with tokenize.open(path) as handle:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(handle.read(), filename=str(path))
    print(f"[source] checked {len(source_integrity['files'])} byte-identical copied files")

    native_sources = [
        ROOT / "disk_detection/training/ops/setup.py",
        ROOT / "disk_detection/training/ops/MultiScaleDeformableAttention.cp312-win_amd64.pyd",
        ROOT / "disk_detection/training/vendor/torch-linear-assignment/setup.py",
    ]
    for path in native_sources:
        if not path.is_file():
            failures.append(f"missing native training component: {path.relative_to(ROOT)}")
    print("[training] original native-extension sources/artifact are present")

    masked_dir = ROOT / "masked_reconstruction"
    sys.path.insert(0, str(masked_dir))
    from dataset import load_pkl

    sample_manifest = json.loads((ROOT / "sample_data" / "manifest.json").read_text(encoding="utf-8"))
    for sample in sample_manifest["files"]:
        path = ROOT / sample["file"]
        if not path.is_file():
            failures.append(f"missing sample: {sample['file']}")
            continue
        if sha256(path) != sample["sha256"]:
            failures.append(f"sample hash mismatch: {sample['file']}")
        payload = load_pkl(path)
        if "diffraction_pattern" not in payload and "diffraction_patterns" not in payload:
            failures.append(f"missing diffraction array: {sample['file']}")
    print(f"[samples] checked {len(sample_manifest['files'])} real-data files")

    disk_dir = ROOT / "disk_detection"
    sys.path.insert(0, str(disk_dir))
    from configs import MODEL_CONFIGS
    from model import SupervisedDNSR

    disk_state = torch.load(disk_dir / "r4_weights.pth", map_location="cpu", weights_only=True)
    disk_model = SupervisedDNSR(MODEL_CONFIGS["r4"])
    disk_model.load_state_dict(disk_state, strict=True)
    print("[model] r4 disk detector: strict load passed")
    del disk_model, disk_state

    from models import SmallUNet

    checkpoint = torch.load(masked_dir / "mixed_domain_epoch48.pth", map_location="cpu", weights_only=False)
    masked_model = SmallUNet(in_channels=2, out_channels=1, base_channels=16)
    masked_model.load_state_dict(checkpoint["model_state"], strict=True)
    print(f"[model] masked U-Net epoch {checkpoint['best_epoch']}: strict load passed")

    if failures:
        raise RuntimeError("Package verification failed:\n- " + "\n- ".join(failures))
    print("Standalone baseline package verification passed.")


if __name__ == "__main__":
    main()
