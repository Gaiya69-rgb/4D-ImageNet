import os
import random
import time
from collections import defaultdict
from itertools import cycle

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

from diffraction_transform import *
from disk_detector_dataset import BraggDataset
from disk_detector_model import SupervisedDNSR

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weight_folder": "./weight/disk_detector",
    "model_version": "r4",

    "epochs": 150,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "warmup_epochs": 5,
    "gradient_clip_val": 0.1,
    "batch_size_sim": 40,
    "batch_size_exp": 5,
    "train_ratio": 0.9,

    "sim_dataset_path_list": ["./dataset/sim_dataset_test"],
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
        "node_count": 384,
        "num_classes": 1,
        "backbone_name": "convnextv2_tiny",
        "backbone_in_channel_count": 1
    },

    "loss_weights": {
        'loss_ce': 2.0,
        'loss_coord': 5.0,
        'loss_center_beam': 1.0,
    }
}

sim_transform = v2.Compose([
    DiffractionSimTransform(min_disk_contrast=0.01, disk_cutoff_count=CONFIG["model_config"]["disk_cutoff_count"])
]).cuda()

exp_transform = v2.Compose([
    DiffractionExpTransform(min_disk_contrast=0.01, disk_cutoff_count=CONFIG["model_config"]["disk_cutoff_count"])
]).cuda()

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_coord: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_coord = outputs["pred_coords"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_coord = torch.cat([v["coords"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_coord = torch.cdist(out_coord, tgt_coord, p=1)
        C = self.cost_coord * cost_coord + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["coords"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# class GPUHungarianMatcher(nn.Module):
#     def __init__(self, cost_class: float = 1, cost_coord: float = 1):
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_coord = cost_coord
#         self.LARGE_COST = 1e6
#
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         bs, num_queries = outputs["pred_logits"].shape[:2]
#         device = outputs["pred_logits"].device
#         out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
#         out_coord = outputs["pred_coords"].flatten(0, 1)
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_coord = torch.cat([v["coords"] for v in targets])
#         cost_class = -out_prob[:, tgt_ids]
#         cost_coord = torch.cdist(out_coord, tgt_coord, p=1)
#         C = self.cost_coord * cost_coord + self.cost_class * cost_class
#         C = C.view(bs, num_queries, -1)
#         sizes = [len(v["coords"]) for v in targets]
#         max_size = max(sizes) if sizes else 0
#         if max_size == 0:
#             empty = torch.empty(0, dtype=torch.int64, device=device)
#             return [(empty, empty) for _ in range(bs)]
#         C_batch = torch.full((bs, num_queries, max_size), self.LARGE_COST, device=device, dtype=torch.float32)
#         C_splits = C.split(sizes, dim=-1)
#         for i, size in enumerate(sizes):
#             if size > 0:
#                 C_batch[i, :, :size] = C_splits[i][i].float()
#         assignment = batch_linear_assignment(C_batch)
#         indices = []
#         for i, size in enumerate(sizes):
#             row_ind = torch.arange(num_queries, device=device)
#             col_ind = assignment[i]
#             valid_mask = (col_ind >= 0) & (col_ind < size)
#             indices.append((row_ind[valid_mask], col_ind[valid_mask]))
#         return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_coords):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_coords(self, outputs, targets, indices, num_coords):
        idx = self._get_src_permutation_idx(indices)
        src_coords = outputs['pred_coords'][idx]
        target_coords = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if num_coords == 0:
            return {'loss_coord': torch.tensor(0.0, device=src_coords.device)}
        loss_coord = F.l1_loss(src_coords, target_coords, reduction='none')
        return {'loss_coord': loss_coord.sum() / num_coords}

    def loss_center_beam(self, outputs, targets, indices, num_coords):
        pred_center_beam = outputs['pred_center_beam']
        target_center_beam = torch.stack([t['center_beam'] for t in targets], dim=0)
        loss_center_beam = F.l1_loss(pred_center_beam, target_center_beam, reduction='mean')
        return {'loss_center_beam': loss_center_beam}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_coords):
        loss_map = {'labels': self.loss_labels, 'coords': self.loss_coords, 'center_beam': self.loss_center_beam}
        return loss_map[loss](outputs, targets, indices, num_coords)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_coords = sum(len(t["labels"]) for t in targets)
        num_coords_tensor = torch.as_tensor([num_coords], dtype=torch.float, device=next(iter(outputs.values())).device)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_coords_tensor))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_coords_tensor)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(CONFIG["seed"])
    output_folder = os.path.join(CONFIG["weight_folder"], CONFIG["model_version"])
    log_dir = os.path.join(output_folder, "log")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    device = torch.device(CONFIG["device"])
    print(f"\nUsing device: {device}")

    full_dataset_sim = BraggDataset(CONFIG, CONFIG["sim_dataset_path_list"])
    full_dataset_exp = BraggDataset(CONFIG, CONFIG["exp_dataset_path_list"])

    n_sim = len(full_dataset_sim)
    n_train_sim = int(CONFIG["train_ratio"] * n_sim)
    n_val_sim = n_sim - n_train_sim
    train_ds_sim, val_ds_sim = random_split(full_dataset_sim, [n_train_sim, n_val_sim])

    n_exp = len(full_dataset_exp)
    n_train_exp = int(CONFIG["train_ratio"] * n_exp)
    n_val_exp = n_exp - n_train_exp
    train_ds_exp, val_ds_exp = random_split(full_dataset_exp, [n_train_exp, n_val_exp])

    train_loader_sim = DataLoader(train_ds_sim, batch_size=CONFIG["batch_size_sim"], shuffle=True, num_workers=0)
    train_loader_exp = DataLoader(train_ds_exp, batch_size=CONFIG["batch_size_exp"], shuffle=True, num_workers=0)

    val_loader_sim = DataLoader(val_ds_sim, batch_size=CONFIG["batch_size_sim"], shuffle=False, num_workers=0)
    val_loader_exp = DataLoader(val_ds_exp, batch_size=CONFIG["batch_size_exp"], shuffle=False, num_workers=0)

    train_loader_exp_iter = cycle(train_loader_exp)
    val_loader_exp_iter = cycle(val_loader_exp)

    print(f"Sim Data: {len(train_ds_sim)} train, {len(val_ds_sim)} val.")
    print(f"Exp Data: {len(train_ds_exp)} train, {len(val_ds_exp)} val.")

    model = SupervisedDNSR(CONFIG["model_config"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SupervisedDNSR model initialized with {total_params:,} trainable parameters.")

    matcher = HungarianMatcher(cost_class=CONFIG['loss_weights']['loss_ce'], cost_coord=CONFIG['loss_weights']['loss_coord'])
    weight_dict = {}
    weight_dict.update({
        'loss_ce': CONFIG['loss_weights']['loss_ce'], 
        'loss_coord': CONFIG['loss_weights']['loss_coord'],
        'loss_center_beam': CONFIG['loss_weights']['loss_center_beam']
    })
    for i in range(CONFIG["model_config"]['decoder_layer_count']):
        weight_dict.update({
            f'loss_ce_{i}': CONFIG['loss_weights']['loss_ce'], 
            f'loss_coord_{i}': CONFIG['loss_weights']['loss_coord'],
            f'loss_center_beam_{i}': CONFIG['loss_weights']['loss_center_beam']
        })
    criterion = SetCriterion(num_classes=CONFIG['model_config']['num_classes'], matcher=matcher, weight_dict=weight_dict, losses=['labels', 'coords', 'center_beam'])
    criterion.to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    num_training_steps = CONFIG["epochs"] * len(train_loader_sim)
    num_warmup_steps = CONFIG["warmup_epochs"] * len(train_loader_sim)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    start_time = time.time()
    best_val_loss = float('inf')

    print("\n--- Starting Training ---")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        criterion.train()

        train_loss_accumulator = defaultdict(float)
        train_loss_accumulator['center_beam'] = 0.0
        progress_bar = tqdm(train_loader_sim, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for sim_images, sim_disks in progress_bar:
            exp_images, exp_disks = next(train_loader_exp_iter)
            sim_images = sim_images.to(device).float()
            sim_disks = sim_disks.to(device)
            exp_images = exp_images.to(device).float()
            exp_disks = exp_disks.to(device)
            with torch.no_grad():
                sim_images_aug, sim_disks_aug, sim_center_beam = sim_transform(sim_images, sim_disks)
                exp_images_aug, exp_disks_aug, exp_center_beam = exp_transform(exp_images, exp_disks)

            image_batch = torch.cat([sim_images_aug, exp_images_aug], dim=0)
            disk_batch = torch.cat([sim_disks_aug, exp_disks_aug], dim=0)
            center_beam_batch = torch.cat([sim_center_beam, exp_center_beam], dim=0)

            target_batch = []
            for i in range(disk_batch.shape[0]):
                valid_mask = disk_batch[i, :, 0] != -1
                valid_coords = disk_batch[i][valid_mask]
                valid_labels = torch.zeros(valid_coords.shape[0], dtype=torch.long, device=device)
                target_batch.append({
                    'coords': valid_coords,
                    'labels': valid_labels,
                    'center_beam': center_beam_batch[i]
                })
            optimizer.zero_grad(set_to_none=True)
            outputs_list = model(image_batch)
            outputs_dict = {
                'pred_logits': outputs_list[-1]['pred_logits'],
                'pred_coords': outputs_list[-1]['pred_coords'],
                'pred_center_beam': outputs_list[-1]['pred_center_beam'],
                'aux_outputs': [{
                    'pred_logits': o['pred_logits'],
                    'pred_coords': o['pred_coords'],
                    'pred_center_beam': o['pred_center_beam']
                } for o in outputs_list[:-1]]
            }
            loss_dict = criterion(outputs_dict, target_batch)
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            total_loss.backward()
            if CONFIG["gradient_clip_val"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_val"])
            optimizer.step()
            scheduler.step()

            train_loss_accumulator['total'] += total_loss.item()
            train_loss_accumulator['ce'] += loss_dict['loss_ce'].item()
            train_loss_accumulator['coord'] += loss_dict['loss_coord'].item()
            train_loss_accumulator['center_beam'] += loss_dict['loss_center_beam'].item()
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                ce=f"{loss_dict['loss_ce'].item():.4f}",
                coord=f"{loss_dict['loss_coord'].item():.4f}",
                cb=f"{loss_dict['loss_center_beam'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

        avg_train_loss = train_loss_accumulator['total'] / len(train_loader_sim)
        writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_CE', train_loss_accumulator['ce'] / len(train_loader_sim), epoch)
        writer.add_scalar('Loss/Train_Coord', train_loss_accumulator['coord'] / len(train_loader_sim), epoch)
        writer.add_scalar('Loss/Train_CenterBeam', train_loss_accumulator['center_beam'] / len(train_loader_sim), epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

        model.eval()
        criterion.eval()

        val_loss_accumulator = defaultdict(float)
        val_loss_accumulator['center_beam'] = 0.0

        with torch.no_grad():
            val_progress_bar = tqdm(val_loader_sim, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]")
            for sim_images, sim_disks in val_progress_bar:
                exp_images, exp_disks = next(val_loader_exp_iter)
                sim_images = sim_images.to(device).float()
                sim_disks = sim_disks.to(device)
                exp_images = exp_images.to(device).float()
                exp_disks = exp_disks.to(device)
                with torch.no_grad():
                    sim_images_aug, sim_disks_aug, sim_center_beam = sim_transform(sim_images, sim_disks)
                    exp_images_aug, exp_disks_aug, exp_center_beam = exp_transform(exp_images, exp_disks)

                image_batch = torch.cat([sim_images_aug, exp_images_aug], dim=0)
                disk_batch = torch.cat([sim_disks_aug, exp_disks_aug], dim=0)
                center_beam_batch = torch.cat([sim_center_beam, exp_center_beam], dim=0)

                target_batch = []
                for i in range(disk_batch.shape[0]):
                    valid_mask = disk_batch[i, :, 0] != -1
                    valid_coords = disk_batch[i][valid_mask]
                    valid_labels = torch.zeros(valid_coords.shape[0], dtype=torch.long, device=device)
                    target_batch.append({
                        'coords': valid_coords,
                        'labels': valid_labels,
                        'center_beam': center_beam_batch[i]
                    })
                outputs_list = model(image_batch)
                outputs_dict = {
                    'pred_logits': outputs_list[-1]['pred_logits'],
                    'pred_coords': outputs_list[-1]['pred_coords'],
                    'pred_center_beam': outputs_list[-1]['pred_center_beam'],
                    'aux_outputs': [{
                        'pred_logits': o['pred_logits'], 
                        'pred_coords': o['pred_coords'],
                        'pred_center_beam': o['pred_center_beam']
                    } for o in outputs_list[:-1]]
                }
                loss_dict = criterion(outputs_dict, target_batch)
                total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                val_loss_accumulator['total'] += total_loss.item()
                val_loss_accumulator['ce'] += loss_dict['loss_ce'].item()
                val_loss_accumulator['coord'] += loss_dict['loss_coord'].item()
                val_loss_accumulator['center_beam'] += loss_dict['loss_center_beam'].item()
                val_progress_bar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    ce=f"{loss_dict['loss_ce'].item():.4f}",
                    coord=f"{loss_dict['loss_coord'].item():.4f}",
                    cb=f"{loss_dict['loss_center_beam'].item():.4f}"
                )

        avg_val_loss = val_loss_accumulator['total'] / len(val_loader_sim)
        writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
        writer.add_scalar('Loss/Val_CE', val_loss_accumulator['ce'] / len(val_loader_sim), epoch)
        writer.add_scalar('Loss/Val_Coord', val_loss_accumulator['coord'] / len(val_loader_sim), epoch)
        writer.add_scalar('Loss/Val_CenterBeam', val_loss_accumulator['center_beam'] / len(val_loader_sim), epoch)

        print(f"Epoch {epoch + 1} Summary | Avg Train Loss: {avg_train_loss:.5f} | Avg Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_folder, f"{CONFIG['model_version']}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"** New best model saved to {save_path} **")

    writer.close()
    total_time = time.time() - start_time
    print(f"\n--- Training Finished in {total_time / 60:.2f} minutes ---")


if __name__ == '__main__':
    main()