import os
import random
import time
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2
from tqdm import tqdm

# Import your custom modules
from diffraction_transform import *
from disk_detector_dataset import BraggDataset
from disk_detector_model import SupervisedDNSR

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weight_folder": "./weight/disk_detector",
    "model_version": "r2",

    "result_folder": "./result/disk_detector",
    "confidence_threshold": 0.5,
    "sim_visualization_count": 50,
    "exp_visualization_count": 50,

    "visualize_intermediate_layers": True,

    "batch_size": 1,
    "train_ratio": 0.9,
    "sim_dataset_path_list": ["./dataset/sim_dataset_v2"],
    "exp_dataset_path_list": ["./dataset/exp_dataset_v1"],
    "disk_count": 256,

    "model_config": {
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
        "backbone_in_channel_count": 1
    },
}

sim_transform = v2.Compose([
    DiffractionSimTransform(min_disk_contrast=0.01, disk_cutoff_count=CONFIG["model_config"]["disk_cutoff_count"])
])

exp_transform = v2.Compose([
    DiffractionExpTransform(min_disk_contrast=0.01, disk_cutoff_count=CONFIG["model_config"]["disk_cutoff_count"])
])


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def visualize_and_save(image, all_layer_preds, gt_coords, save_path, idx, pred_center_beam=None, gt_center_beam=None):
    """
    all_layer_preds: List of tuples [(coords, scores), (coords, scores), ...]
                     representing predictions for each layer.
    pred_center_beam: [2] array with normalized (x, y) center beam prediction
    gt_center_beam: [2] array with normalized (x, y) center beam ground truth
    """
    H, W = image.shape

    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')

    # 1. Plot Predictions (Gradient Color)
    num_layers = len(all_layer_preds)

    # Select colormap (Viridis: Purple->Yellow)
    # We avoid Red because that is for GT.
    cmap = plt.get_cmap('viridis')

    if CONFIG["visualize_intermediate_layers"] and num_layers > 1:
        # Plot layers 0 to N-1
        for layer_i, (coords, scores) in enumerate(all_layer_preds):
            if len(coords) == 0:
                continue

            # Normalize layer index from 0.0 to 1.0 for color selection
            color_val = layer_i / (num_layers - 1)
            color = cmap(color_val)

            is_final_layer = (layer_i == num_layers - 1)

            if is_final_layer:
                # Final Layer: Prominent 'x'
                plt.scatter(coords[:, 0] * W, coords[:, 1] * H,
                            color=color, marker='x', s=100, linewidth=2,
                            label=f'Final Pred (L{layer_i})', zorder=10)

                # Annotate scores only for final layer
                for k, (x, y) in enumerate(coords):
                    plt.text(x * W + 5, y * H + 5, f"{scores[k]:.2f}",
                             color=color, fontsize=9, fontweight='bold')
            else:
                # Intermediate Layers: Smaller dots, slightly transparent
                plt.scatter(coords[:, 0] * W, coords[:, 1] * H,
                            color=color, marker='.', s=50, alpha=0.7,
                            zorder=5)  # No label to avoid clutter, or label only first
                if layer_i == 0:
                    # Dummy plot for legend
                    plt.plot([], [], color=cmap(0.0), marker='.', linestyle='None', label='Layer 0 (Start)')

    else:
        # Simple mode: Just plot the last one as Red 'x' (if not using gradient)
        # But prompt asked for GT as red, so let's keep Preds as distinct color (Cyan) if mode is off
        last_coords, last_scores = all_layer_preds[-1]
        if len(last_coords) > 0:
            plt.scatter(last_coords[:, 0] * W, last_coords[:, 1] * H,
                        c='cyan', marker='x', s=80, linewidth=1.5, label='Predicted')

    # 2. Plot Ground Truth (Red)
    if len(gt_coords) > 0:
        plt.scatter(gt_coords[:, 0] * W, gt_coords[:, 1] * H,
                    c='red', marker='+', s=120, linewidth=2.0, label='Ground Truth', zorder=15)

    # 3. Plot Center Beam
    if pred_center_beam is not None:
        plt.scatter(pred_center_beam[0] * W, pred_center_beam[1] * H,
                    c='yellow', marker='*', s=300, linewidth=2.0, 
                    label='Center Beam (Pred)', zorder=20, edgecolors='black')
    if gt_center_beam is not None:
        plt.scatter(gt_center_beam[0] * W, gt_center_beam[1] * H,
                    c='orange', marker='*', s=300, linewidth=2.0,
                    label='Center Beam (GT)', zorder=20, edgecolors='black')

    plt.title(f"Val Sample #{idx} | Layers: {num_layers}")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def process_dataset(dataset, transform, dataset_type, num_samples, model, device, output_dir):
    """Process a single dataset type (sim or exp)"""
    n_samples = len(dataset)
    n_train = int(CONFIG["train_ratio"] * n_samples)
    n_val = n_samples - n_train

    _, val_dataset = random_split(dataset, [n_train, n_val])

    # Sampling
    total_val = len(val_dataset)
    num_to_visualize = min(num_samples, total_val)
    viz_indices = random.sample(range(total_val), num_to_visualize)
    viz_dataset = Subset(val_dataset, viz_indices)
    viz_loader = DataLoader(viz_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    print(f"\nProcessing {dataset_type} dataset: {num_to_visualize} samples")

    with torch.no_grad():
        for batch_i, (image_batch, disk_batch) in enumerate(tqdm(viz_loader, desc=f"Inference [{dataset_type}]")):
            image_batch = image_batch.to(device).float()
            disk_batch = disk_batch.to(device)
            image_batch, disk_batch, center_beam_gt = transform(image_batch, disk_batch)

            # outputs_list contains output dictionaries from all decoder layers
            outputs_list = model(image_batch)

            # Process single batch item (batch_size=1)
            i = 0

            global_ptr = batch_i * CONFIG["batch_size"] + i
            if global_ptr >= len(viz_indices): break
            original_val_idx = viz_indices[global_ptr]

            # Collect predictions from ALL layers
            all_layer_preds = []

            layers_to_process = outputs_list if CONFIG["visualize_intermediate_layers"] else [outputs_list[-1]]

            # Extract center beam prediction from final layer
            pred_center_beam = outputs_list[-1]['pred_center_beam'][i].cpu().numpy()  # [2]
            
            for layer_out in layers_to_process:
                pred_logits = layer_out['pred_logits'][i]  # [Num_Queries, Classes+1]
                pred_coords = layer_out['pred_coords'][i]  # [Num_Queries, 2]

                probas = pred_logits.softmax(-1)[:, :-1]
                scores, _ = probas.max(-1)

                keep = scores > CONFIG["confidence_threshold"]

                valid_coords = pred_coords[keep].cpu().numpy()
                valid_scores = scores[keep].cpu().numpy()

                all_layer_preds.append((valid_coords, valid_scores))

            # GT Processing
            valid_mask = disk_batch[i, :, 0] != -1
            valid_gt_coords = disk_batch[i][valid_mask][:, :2].cpu().numpy()
            center_beam_gt_np = center_beam_gt[i].cpu().numpy()

            img_np = image_batch[i, 0].cpu().numpy()

            save_name = os.path.join(output_dir, f"{dataset_type}_sample_{original_val_idx:04d}_gradient.png")

            visualize_and_save(
                img_np,
                all_layer_preds,
                valid_gt_coords,
                save_name,
                original_val_idx,
                pred_center_beam=pred_center_beam,
                gt_center_beam=center_beam_gt_np
            )


def main():
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])

    weight_path = os.path.join(CONFIG["weight_folder"], CONFIG["model_version"], f"{CONFIG['model_version']}_best.pth")
    output_dir = os.path.join(CONFIG["result_folder"], CONFIG["model_version"])

    if os.path.exists(output_dir):
        print(f"Cleaning existing result folder: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading weights from: {weight_path}")
    print(f"Saving results to: {output_dir}")

    # 1. Dataset Setup
    print("\n--- Loading Datasets ---")
    full_dataset_sim = BraggDataset(CONFIG, CONFIG["sim_dataset_path_list"])
    full_dataset_exp = BraggDataset(CONFIG, CONFIG["exp_dataset_path_list"])
    
    print(f"Sim dataset: {len(full_dataset_sim)} samples")
    print(f"Exp dataset: {len(full_dataset_exp)} samples")

    # 2. Model
    model = SupervisedDNSR(CONFIG["model_config"]).to(device)
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"\nWeights loaded successfully from {weight_path}")
    except FileNotFoundError:
        print(f"Error: Weight file not found at {weight_path}")
        return
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    print("\n--- Starting Inference ---")

    # 3. Process Sim Dataset
    process_dataset(
        full_dataset_sim,
        sim_transform,
        "sim",
        CONFIG["sim_visualization_count"],
        model,
        device,
        output_dir
    )

    # 4. Process Exp Dataset
    process_dataset(
        full_dataset_exp,
        exp_transform,
        "exp",
        CONFIG["exp_visualization_count"],
        model,
        device,
        output_dir
    )

    print("\nInference complete.")


if __name__ == '__main__':
    main()