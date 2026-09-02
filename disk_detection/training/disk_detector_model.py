import math
from typing import Dict, List, Any

import timm
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from ops.functions.ms_deform_attn_func import MSDeformAttnFunction


class BackboneEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet34', backbone_in_channel_count: int = 1, token_depth: int = 256, out_indices: tuple = None):
        super().__init__()
        if out_indices is None:
            if 'resnet' in model_name:
                out_indices = (1, 2, 3, 4)
            else:
                out_indices = (0, 1, 2, 3)
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            in_chans=backbone_in_channel_count,
            out_indices=out_indices,
        )
        feature_info = self.encoder.feature_info.channels()
        self.input_projection_list = nn.ModuleList()
        for in_channels in feature_info:
            self.input_projection_list.append(
                nn.Conv2d(in_channels, token_depth, kernel_size=1)
            )
        print(f"Initialized FPN-style backbone from '{model_name}'.")
        print(f"> Multi-scale features from stages with channels: {feature_info}")
        print(f"> All projected to token_depth: {token_depth}")

    def forward(self, x: torch.Tensor) -> List[Tensor]:
        feature = self.encoder(x)
        feature_projected_list = [
            proj(feat) for proj, feat in zip(self.input_projection_list, feature)
        ]
        return feature_projected_list


class ScalarPositionalEncoding(nn.Module):
    def __init__(self, frequency_count: int, include_input: bool = True):
        super().__init__()
        self.frequency_count = frequency_count
        self.include_input = include_input
        self.register_buffer(
            'frequency_list',
            2.0 ** torch.arange(frequency_count),
            persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x_projected = x.unsqueeze(-1) * self.frequency_list * torch.pi
        encoding = torch.cat([torch.sin(x_projected), torch.cos(x_projected)], dim=-1)
        encoding = encoding.flatten(start_dim=-2)
        if self.include_input:
            encoding = torch.cat([x, encoding], dim=-1)
        return encoding


class NodePositionalEncoder(nn.Module):
    def __init__(self, token_depth: int):
        super().__init__()
        frequency_count = 10
        scalar_encoding_dim = 2 * frequency_count
        self.projection = nn.Linear(2 * scalar_encoding_dim, token_depth)
        self.encoder = ScalarPositionalEncoding(frequency_count=frequency_count, include_input=False)
        print(f"Initialized NeRF-style Positional Encoding with {frequency_count} frequency bands.")
        print(f"> Projecting from {2 * scalar_encoding_dim} dims to token_depth: {token_depth}")

    def forward(self, node_coordinate_list: Tensor) -> Tensor:
        encoding_x_list = self.encoder(node_coordinate_list[..., 0:1])
        encoding_y_list = self.encoder(node_coordinate_list[..., 1:2])
        concatenated_encodings = torch.cat([encoding_x_list, encoding_y_list], dim=-1)
        return self.projection(concatenated_encodings)


class MultiScalePositionalEncoder(nn.Module):
    def __init__(self, token_depth: int, feature_shape_list: List[Dict[str, int]]):
        super().__init__()
        self.encoding_list = nn.ParameterList()
        for shape in feature_shape_list:
            h, w = shape['height'], shape['width']
            param = nn.Parameter(torch.randn(1, token_depth, h, w))
            self.encoding_list.append(param)
        print(f"Initialized Multi-scale Positional Encodings for feature maps of sizes:")
        for shape in feature_shape_list:
            print(f"> {shape['height']}x{shape['width']}")

    def forward(self, feature_list: List[Tensor]) -> List[Tensor]:
        if len(feature_list) != len(self.encoding_list):
            raise ValueError("Number of feature maps does not match number of positional encodings.")
        return [
            feature + encoding for feature, encoding in zip(feature_list, self.encoding_list)
        ]


class MSDeformAttn(nn.Module):
    def __init__(self, token_depth, n_levels, n_heads, n_points):
        super().__init__()
        self.im2col_step = 128
        self.token_depth = token_depth
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(token_depth, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(token_depth, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(token_depth, token_depth)
        self.output_proj = nn.Linear(token_depth, token_depth)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        B, Nq, C = query.shape
        B, N_in, C = input_flatten.shape
        value = self.value_proj(input_flatten).view(B, N_in, self.n_heads, C // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(B, Nq, self.n_heads, self.n_levels * self.n_points).softmax(-1)
        attention_weights = attention_weights.view(B, Nq, self.n_heads, self.n_levels, self.n_points)
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        reference_points_reshaped = reference_points.view(B, Nq, 1, 1, 1, 2)
        sampling_locations = reference_points_reshaped + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations,
            attention_weights, self.im2col_step
        )
        output = self.output_proj(output)
        return output


class MSDeformAttnEncoderLayer(nn.Module):
    def __init__(self, token_depth, attention_head_count, n_levels, n_points):
        super().__init__()
        self.self_attn = MSDeformAttn(token_depth, n_levels, attention_head_count, n_points)
        self.ffn = nn.Sequential(
            nn.Linear(token_depth, token_depth * 4), nn.ReLU(),
            nn.Linear(token_depth * 4, token_depth)
        )
        self.norm1 = nn.LayerNorm(token_depth)
        self.norm2 = nn.LayerNorm(token_depth)

    def forward(self, src, ref_points, spatial_shapes, level_start_index):
        src2 = self.self_attn(src, ref_points, src, spatial_shapes, level_start_index)
        src = self.norm1(src + src2)
        src2 = self.ffn(src)
        src = self.norm2(src + src2)
        return src


class MSDeformAttnDecoderLayer(nn.Module):
    def __init__(self, token_depth, attention_head_count, n_levels, n_points):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(token_depth, attention_head_count, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(token_depth)
        self.cross_attn = MSDeformAttn(token_depth, n_levels, attention_head_count, n_points)
        self.norm2 = nn.LayerNorm(token_depth)
        self.ffn = nn.Sequential(
            nn.Linear(token_depth, token_depth * 4), nn.ReLU(),
            nn.Linear(token_depth * 4, token_depth)
        )
        self.norm3 = nn.LayerNorm(token_depth)

    def forward(self, tgt, memory, ref_points, spatial_shapes, level_start_index):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt, ref_points, memory, spatial_shapes, level_start_index)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt


class SupervisedDNSR(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.node_count = config["node_count"]
        self.token_depth = config["token_depth"]
        self.num_classes = config["num_classes"]
        token_depth = self.token_depth
        num_classes = self.num_classes
        attention_head_count = config["attention_head_count"]
        encoder_layer_count = config.get("encoder_layer_count", 6)
        decoder_layer_count = config.get("decoder_layer_count", 6)
        n_levels = config.get("num_feature_levels", 4)
        n_points = config.get("deformable_attention_points", 4)

        self.encoder = BackboneEncoder(
            model_name=config.get("backbone_name", "resnet34"),
            backbone_in_channel_count=config.get("backbone_in_channel_count", 1),
            token_depth=token_depth
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, config.get("backbone_in_channel_count", 1), 256, 256)
            dummy_feature_list = self.encoder(dummy_input)
            feature_shape_list = [{'height': f.shape[2], 'width': f.shape[3]} for f in dummy_feature_list]
        self.feature_positional_encoder = MultiScalePositionalEncoder(token_depth, feature_shape_list)
        self.node_positional_encoder = NodePositionalEncoder(token_depth)

        self.encoder_layer_list = nn.ModuleList([
            MSDeformAttnEncoderLayer(token_depth, attention_head_count, n_levels, n_points)
            for _ in range(encoder_layer_count)
        ])
        print(f"Initialized Deformable Attention Encoder with {encoder_layer_count} layers.")

        self.enc_output_norm = nn.LayerNorm(token_depth)
        self.enc_score_head = nn.Linear(token_depth, num_classes)
        self.enc_coord_head = nn.Linear(token_depth, 2)

        self.center_beam_query = nn.Embedding(1, token_depth)
        self.query_tokens = nn.Embedding(self.node_count, token_depth)
        print(f"Initialized learned query tokens: {self.node_count} tokens with dimension {token_depth}.")

        self.decoder_layer_list = nn.ModuleList([
            MSDeformAttnDecoderLayer(token_depth, attention_head_count, n_levels, n_points)
            for _ in range(decoder_layer_count)
        ])
        print(f"Initialized Deformable Attention Decoder with {decoder_layer_count} layers.")

        self.coordinate_head = nn.ModuleList([nn.Linear(token_depth, 2) for _ in range(decoder_layer_count)])
        self.class_head = nn.ModuleList([nn.Linear(token_depth, num_classes + 1) for _ in range(decoder_layer_count)])  # Add 1 for "no object" class

        self.center_beam_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_depth, token_depth),
                nn.ReLU(),
                nn.Linear(token_depth, 2),
                nn.Sigmoid()
            ) for _ in range(decoder_layer_count)
        ])

        self._init_weights()

    def _get_ref_points(self, spatial_shapes, device):
        ref_points_list = []
        for H, W in spatial_shapes:
            y, x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = y.flatten().unsqueeze(0) / H
            ref_x = x.flatten().unsqueeze(0) / W
            ref_points = torch.stack((ref_x, ref_y), -1).squeeze(0)
            ref_points_list.append(ref_points)
        return torch.cat(ref_points_list, 0)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        prob_no_object = 0.99
        bias_value = -math.log((1 - prob_no_object) / prob_no_object)
        for head in self.class_head:
            nn.init.constant_(head.bias, bias_value)
        print(f"Prediction head biases initialized. Initial 'no object' probability: {prob_no_object:.2f}")

    def _gen_encoder_coordinate_grid(self, spatial_shapes, device):
        ref_points_list = []
        for H, W in spatial_shapes:
            y, x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = y.flatten() / H
            ref_x = x.flatten() / W
            ref_points = torch.stack((ref_x, ref_y), -1)
            ref_points_list.append(ref_points)
        return torch.cat(ref_points_list, 0)

    def forward(self, x: Tensor) -> List[Dict[str, Tensor]]:
        batch_size = x.shape[0]
        token_depth = self.token_depth
        num_classes = self.num_classes

        feature_list = self.encoder(x)
        feature_encoded_list = self.feature_positional_encoder(feature_list)
        spatial_shapes = torch.as_tensor([(f.shape[2], f.shape[3]) for f in feature_encoded_list], dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        memory_list = [feature.flatten(2).permute(0, 2, 1) for feature in feature_encoded_list]
        memory = torch.cat(memory_list, dim=1)
        ref_points_encoder = self._get_ref_points(spatial_shapes, x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        for layer in self.encoder_layer_list:
            memory = layer(memory, ref_points_encoder, spatial_shapes, level_start_index)

        base_grid = self._gen_encoder_coordinate_grid(spatial_shapes, x.device)
        memory_norm = self.enc_output_norm(memory)
        enc_scores = self.enc_score_head(memory_norm)
        enc_deltas = self.enc_coord_head(memory_norm)
        base_grid_batch = base_grid.unsqueeze(0)
        eps = 1e-4
        base_grid_clamped = base_grid_batch.clamp(eps, 1 - eps)
        base_grid_logit = torch.log(base_grid_clamped / (1 - base_grid_clamped))
        enc_pred_coords = torch.sigmoid(base_grid_logit + enc_deltas)
        objectness = enc_scores.sigmoid().max(dim=-1)[0]
        topk_indices = objectness.topk(self.node_count, dim=1)[1]
        topk_ref_points = enc_pred_coords.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, 2))
        topk_ref_points = topk_ref_points.detach()
        
        # Use learned query tokens instead of memory features
        learned_query_tokens = self.query_tokens.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        topk_enc_scores = enc_scores.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, num_classes))
        no_object_logit = torch.full((batch_size, self.node_count, 1), -10.0, device=x.device)
        enc_pred_logits = torch.cat([topk_enc_scores, no_object_logit], dim=-1)
        enc_center_beam = self.center_beam_head[0](memory.mean(dim=1))
        
        enc_output_dict = {
            "pred_logits": enc_pred_logits,
            "pred_coords": topk_ref_points,
            "pred_center_beam": enc_center_beam
        }

        center_query = self.center_beam_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        center_ref_point = torch.tensor([[0.5, 0.5]], device=x.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = torch.cat([center_query, learned_query_tokens], dim=1)
        ref_points = torch.cat([center_ref_point, topk_ref_points], dim=1)

        intermediate_outputs = []
        for i, layer in enumerate(self.decoder_layer_list):
            query_pos = self.node_positional_encoder(ref_points)
            q = tgt + query_pos
            tgt = layer(
                tgt=q,
                memory=memory,
                ref_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            pred_center_beam = self.center_beam_head[i](tgt[:, 0])
            pred_logits = self.class_head[i](tgt[:, 1:])

            pred_coords_delta = self.coordinate_head[i](tgt[:, 1:])
            ref_points_disk = ref_points[:, 1:]

            eps = 1e-4
            ref_points_clamped = ref_points_disk.clamp(eps, 1 - eps)
            ref_points_logit = torch.log(ref_points_clamped / (1 - ref_points_clamped))
            pred_coords = torch.sigmoid(ref_points_logit + pred_coords_delta)

            ref_points = torch.cat([center_ref_point, pred_coords.detach()], dim=1)
            
            intermediate_outputs.append({
                "pred_logits": pred_logits,
                "pred_coords": pred_coords,
                "pred_center_beam": pred_center_beam
            })

        return [enc_output_dict] + intermediate_outputs