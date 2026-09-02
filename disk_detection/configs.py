from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "disk_detection" / "outputs"


MODEL_CONFIGS = {
    "r1": {
        "token_depth": 192,
        "attention_head_count": 8,
        "encoder_layer_count": 4,
        "decoder_layer_count": 6,
        "num_feature_levels": 4,
        "deformable_attention_points": 4,
        "disk_cutoff_count": 64,
        "node_count": 100,
        "num_classes": 1,
        "backbone_name": "resnet34",
        "backbone_in_channel_count": 1,
    },
    "r2": {
        "token_depth": 192,
        "attention_head_count": 8,
        "encoder_layer_count": 4,
        "decoder_layer_count": 6,
        "num_feature_levels": 4,
        "deformable_attention_points": 4,
        "disk_cutoff_count": 64,
        "node_count": 100,
        "num_classes": 1,
        "backbone_name": "resnet34",
        "backbone_in_channel_count": 1,
    },
    "r3": {
        "token_depth": 192,
        "attention_head_count": 8,
        "encoder_layer_count": 4,
        "decoder_layer_count": 6,
        "num_feature_levels": 4,
        "deformable_attention_points": 4,
        "disk_cutoff_count": 64,
        "node_count": 384,
        "num_classes": 1,
        "backbone_name": "convnextv2_tiny",
        "backbone_in_channel_count": 1,
    },
    "r4": {
        "token_depth": 192,
        "attention_head_count": 8,
        "encoder_layer_count": 4,
        "decoder_layer_count": 6,
        "num_feature_levels": 4,
        "deformable_attention_points": 4,
        "disk_cutoff_count": 64,
        "node_count": 384,
        "num_classes": 1,
        "backbone_name": "convnextv2_tiny",
        "backbone_in_channel_count": 1,
    },
}


WEIGHT_PATHS = {
    "r1": PROJECT_ROOT / "disk_detector_weight" / "r1" / "r1_best.pth",
    "r2": PROJECT_ROOT / "disk_detector_weight" / "r2" / "r2_best.pth",
    "r3": PROJECT_ROOT / "disk_detector_weight" / "r3" / "r3_best.pth",
    "r4": PROJECT_ROOT / "disk_detector_weight" / "r4" / "r4_best.pth",
}
