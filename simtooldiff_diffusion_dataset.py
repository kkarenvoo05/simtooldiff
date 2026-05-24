from typing import Dict, Optional, Sequence
import bisect
import copy

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def _as_bool_list(values, length: int):
    if values is None:
        return [False] + [True] * (length - 1)
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value.lower() in ("1", "true", "yes", "on"))
        else:
            result.append(bool(value))
    if len(result) != length:
        raise ValueError(
            f"copy_to_memory has length {len(result)} but zarr_paths has length {length}")
    return result


class SimtoolImageDatasetWithBackend(BaseImageDataset):
    """SimTool zarr dataset with configurable in-memory or disk-backed storage."""

    def __init__(
            self,
            zarr_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            state_dim: int = 29,
            image_shape: Sequence[int] = (3, 96, 96),
            copy_to_memory: bool = True,
            ):
        super().__init__()
        if copy_to_memory:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=['img', 'state', 'action'])
        else:
            self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')

        full_state_dim = self.replay_buffer['state'].shape[1]
        assert 1 <= int(state_dim) <= full_state_dim, (
            f"state_dim={state_dim} not in [1, {full_state_dim}]")
        image_shape = tuple(int(x) for x in image_shape)
        assert len(image_shape) == 3 and image_shape[0] == 3, (
            f"image_shape must be (3, H, W), got {image_shape}")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_dim = int(state_dim)
        self.image_shape = image_shape
        self.copy_to_memory = bool(copy_to_memory)
        self.zarr_path = zarr_path

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :self.state_dim].astype(np.float32)
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32) / 255.0
        _, target_h, target_w = self.image_shape
        if image.shape[-2:] != (target_h, target_w):
            image_t = torch.from_numpy(image)
            image_t = F.interpolate(
                image_t, size=(target_h, target_w),
                mode='bilinear', align_corners=False)
            image = image_t.numpy()
        return {
            'obs': {
                'image': image,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


class ConcatSimtoolImageDataset(BaseImageDataset):
    """Concatenate multiple SimTool zarr datasets for diffusion-policy training.

    This avoids materializing a new clean+anchored zarr for each sweep setting.
    By default the first source is disk-backed and later sources are copied into
    RAM, which keeps the shared clean dataset out of every job's Python heap.
    """

    def __init__(
            self,
            zarr_paths: Sequence[str],
            zarr_path: Optional[str] = None,
            repeat_factors: Optional[Sequence[int]] = None,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            state_dim: int = 29,
            image_shape: Sequence[int] = (3, 96, 96),
            copy_to_memory: Optional[Sequence[bool]] = None,
            ):
        super().__init__()
        if zarr_paths is None or len(zarr_paths) == 0:
            if zarr_path is None:
                raise ValueError("Expected zarr_paths or zarr_path")
            zarr_paths = [zarr_path]

        self.zarr_paths = list(zarr_paths)
        if repeat_factors is None:
            repeat_factors = [1] * len(self.zarr_paths)
        self.repeat_factors = [int(x) for x in repeat_factors]
        if len(self.repeat_factors) != len(self.zarr_paths):
            raise ValueError(
                f"repeat_factors has length {len(self.repeat_factors)} but "
                f"zarr_paths has length {len(self.zarr_paths)}")
        if any(x < 1 for x in self.repeat_factors):
            raise ValueError(f"repeat_factors must be >= 1, got {self.repeat_factors}")

        self.copy_to_memory = _as_bool_list(
            copy_to_memory,
            length=len(self.zarr_paths))

        self.datasets = [
            SimtoolImageDatasetWithBackend(
                zarr_path=path,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                state_dim=state_dim,
                image_shape=image_shape,
                copy_to_memory=copy_flag,
            )
            for path, copy_flag in zip(self.zarr_paths, self.copy_to_memory)
        ]
        self._rebuild_index()
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_dim = int(state_dim)
        self.image_shape = tuple(int(x) for x in image_shape)

    def _rebuild_index(self):
        lengths = []
        for dataset, repeat_factor in zip(self.datasets, self.repeat_factors):
            lengths.append(len(dataset) * repeat_factor)
        self._lengths = lengths
        self._cumulative_lengths = np.cumsum(lengths).astype(np.int64)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.datasets = [dataset.get_validation_dataset() for dataset in self.datasets]
        val_set._rebuild_index()
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action_arrays = []
        agent_pos_arrays = []
        for dataset in self.datasets:
            action_arrays.append(dataset.replay_buffer['action'])
            agent_pos_arrays.append(dataset.replay_buffer['state'][..., :self.state_dim])

        data = {
            'action': np.concatenate(action_arrays, axis=0),
            'agent_pos': np.concatenate(agent_pos_arrays, axis=0),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([
            dataset.replay_buffer['action'] for dataset in self.datasets
        ], axis=0))

    def __len__(self) -> int:
        if len(self._cumulative_lengths) == 0:
            return 0
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        prev_end = 0 if dataset_idx == 0 else int(self._cumulative_lengths[dataset_idx - 1])
        local_idx = idx - prev_end
        dataset = self.datasets[dataset_idx]
        return dataset[local_idx % len(dataset)]
