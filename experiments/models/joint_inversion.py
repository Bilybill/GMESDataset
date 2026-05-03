from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.datasets.gmes_inverse_dataset import AVAILABLE_MODALITIES
from experiments.models.common import ConvBlock3d
from experiments.models.encoders_2d import ObservationEncoder2D
from experiments.models.encoders_mt import MTObservationEncoder
from experiments.models.encoders_seismic import SeismicObservationEncoder


class VolumeDecoder3D(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 4,
        target_shape: tuple[int, int, int] = (64, 64, 64),
        seed_shape: tuple[int, int, int] = (4, 4, 4),
        base_channels: int = 128,
    ):
        super().__init__()
        self.target_shape = tuple(int(x) for x in target_shape)
        self.seed_shape = tuple(int(x) for x in seed_shape)
        self.base_channels = int(base_channels)
        seed_size = self.base_channels * math.prod(self.seed_shape)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, seed_size),
            nn.GELU(),
        )

        max_ratio = max(
            math.ceil(self.target_shape[0] / self.seed_shape[0]),
            math.ceil(self.target_shape[1] / self.seed_shape[1]),
            math.ceil(self.target_shape[2] / self.seed_shape[2]),
        )
        num_upsamples = max(0, math.ceil(math.log2(max(max_ratio, 1))))

        blocks = []
        channels = self.base_channels
        for _ in range(num_upsamples):
            next_channels = max(channels // 2, 16)
            blocks.append(ConvBlock3d(channels, next_channels))
            channels = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.projection(latent)
        x = x.view(latent.shape[0], self.base_channels, *self.seed_shape)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = block(x)
        if tuple(x.shape[-3:]) != self.target_shape:
            x = F.interpolate(x, size=self.target_shape, mode="trilinear", align_corners=False)
        return self.output(x)


class LateFusionJointInversionModel(nn.Module):
    def __init__(
        self,
        modalities: tuple[str, ...] = AVAILABLE_MODALITIES,
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        target_shape: tuple[int, int, int] = (64, 64, 64),
        decoder_base_channels: int = 128,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.embedding_dim = int(embedding_dim)
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            if modality in ("gravity", "magnetic"):
                self.encoders[modality] = ObservationEncoder2D(in_channels=1, embedding_dim=embedding_dim)
            elif modality == "mt":
                self.encoders[modality] = MTObservationEncoder(in_channels=76, embedding_dim=embedding_dim)
            elif modality == "seismic":
                self.encoders[modality] = SeismicObservationEncoder(in_channels=25, embedding_dim=embedding_dim)
            else:
                raise ValueError(f"Unsupported modality '{modality}'.")

        fused_dim = len(self.modalities) * embedding_dim + len(self.modalities)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )
        self.decoder = VolumeDecoder3D(
            latent_dim=fusion_hidden_dim,
            out_channels=4,
            target_shape=target_shape,
            base_channels=decoder_base_channels,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        modality_mask = []
        batch_size = None
        device = None
        dtype = None

        for modality in self.modalities:
            tensor = inputs.get(modality)
            if tensor is not None:
                if batch_size is None:
                    batch_size = int(tensor.shape[0])
                    device = tensor.device
                    dtype = tensor.dtype
                embeddings.append(self.encoders[modality](tensor))
                modality_mask.append(torch.ones(tensor.shape[0], 1, device=tensor.device, dtype=tensor.dtype))
            else:
                if batch_size is None:
                    raise ValueError("At least one modality tensor must be provided.")
                embeddings.append(torch.zeros(batch_size, self.embedding_dim, device=device, dtype=dtype))
                modality_mask.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))

        fused = torch.cat([*embeddings, *modality_mask], dim=1)
        return self.decoder(self.fusion(fused))
