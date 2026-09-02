from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DiffractionPatternDataset, load_pkl
from masks import make_mask
from metrics import masked_mae, masked_mse, psnr_from_mse, weighted_masked_mse
from models import SmallUNet


CHECKPOINT = THIS_DIR / "mixed_domain_epoch48.pth"
SAMPLES = [
    ("experimental", PACKAGE_ROOT / "sample_data" / "experimental" / "Pd_experimental_pattern.pkl"),
    ("simulation", PACKAGE_ROOT / "sample_data" / "simulation" / "CeO2_simulation_pattern.pkl"),
]
OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "masked_reconstruction"
IMAGE_SIZE = 192


def load_image(path: Path) -> torch.Tensor:
    payload = load_pkl(path)
    if "diffraction_pattern" in payload:
        pattern = payload["diffraction_pattern"]
    elif "diffraction_patterns" in payload:
        pattern = payload["diffraction_patterns"][0]
    else:
        raise KeyError(f"No diffraction pattern array found in {path}")
    image = DiffractionPatternDataset.preprocess(pattern)
    if image.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SmallUNet(in_channels=2, out_channels=1, base_channels=16).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(SAMPLES), 4, figsize=(8.2, 4.3), squeeze=False)
    records = []

    for row, (domain, path) in enumerate(SAMPLES):
        image = load_image(path).unsqueeze(0).to(device)
        valid_mask = torch.ones_like(image[:, :1])
        torch.manual_seed(42 + row)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + row)
        mask = make_mask(
            batch_size=1,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            mask_type="spot",
            mask_ratio=0.5,
            patch_size=16,
            device=device,
            images=image,
            valid_mask=valid_mask,
            spot_radius=0.035,
            center_exclude_radius=0.06,
            spot_rank=1,
            peak_window=11,
            spot_selection="random",
            min_spots=1,
            max_spots=4,
            min_peak_fraction=0.12,
        )
        masked = image * (1.0 - mask)
        model_input = torch.cat([masked, mask], dim=1)
        with torch.inference_mode():
            prediction = model(model_input)

        mse = float(masked_mse(prediction, image, mask).cpu())
        mae = float(masked_mae(prediction, image, mask).cpu())
        weighted_mse = float(
            weighted_masked_mse(
                prediction,
                image,
                mask,
                intensity_weight=8.0,
                intensity_gamma=1.0,
            ).cpu()
        )
        records.append(
            {
                "domain": domain,
                "source_file": path.relative_to(PACKAGE_ROOT).as_posix(),
                "masked_pixels": int(mask.sum().item()),
                "masked_mse": mse,
                "masked_mae": mae,
                "weighted_masked_mse": weighted_mse,
                "psnr_db": psnr_from_mse(mse),
            }
        )

        panels = [
            image[0, 0],
            masked[0, 0],
            prediction[0, 0],
            (prediction - image).abs()[0, 0] * mask[0, 0],
        ]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_axis_off()
        axes[row, 0].set_ylabel(domain.capitalize())

    for axis, title in zip(axes[0], ["Target", "Masked input", "Reconstruction", "Masked error"]):
        axis.set_title(title)
    figure.tight_layout(pad=0.5)
    figure.savefig(OUTPUT_DIR / "masked_reconstruction_examples.png", dpi=240)
    plt.close(figure)

    result = {
        "schema_version": "4d-imagenet-packaged-masked-example-v1",
        "checkpoint": CHECKPOINT.relative_to(PACKAGE_ROOT).as_posix(),
        "best_epoch": int(checkpoint["best_epoch"]),
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "mask": {
            "type": "spot",
            "spot_radius_fraction": 0.035,
            "center_exclude_radius_fraction": 0.06,
            "minimum_spots": 1,
            "maximum_spots": 4,
            "seed_base": 42,
        },
        "records": records,
    }
    output_json = OUTPUT_DIR / "masked_reconstruction_examples.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Loaded epoch-{checkpoint['best_epoch']} checkpoint on {device}.")
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
