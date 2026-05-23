import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

isaacgym_module = types.ModuleType("isaacgym")
isaacgym_module.gymapi = object()
sys.modules.setdefault("isaacgym", isaacgym_module)

import deployment.rl_player as rl_player_module
import stage5_anchored_recovery as anchored
import stage5_collect_dataset as stage5_collector
import collect_dataset_pick_place_release as stage6_collector


class _FakeDataset:
    def __init__(self, shape):
        self.shape = shape


class _FakeRoot:
    def __init__(self, n_episodes: int, attrs=None):
        self.attrs = {} if attrs is None else dict(attrs)
        self._meta = {"episode_ends": _FakeDataset((n_episodes,))}

    def __getitem__(self, key):
        if key == "meta":
            return self._meta
        raise KeyError(key)


def test_sample_branch_trigger_steps_stay_in_bounds_and_respect_gap():
    rng = np.random.default_rng(0)
    trigger_steps = anchored.sample_branch_trigger_steps(
        rng=rng,
        branch_min_step=10,
        branch_max_step=40,
        branches_per_rollout=4,
        min_gap=6,
    )

    assert trigger_steps == sorted(trigger_steps)
    assert len(trigger_steps) <= 4
    assert all(10 <= step <= 40 for step in trigger_steps)
    assert all(
        later - earlier >= 6
        for earlier, later in zip(trigger_steps, trigger_steps[1:])
    )


def test_sample_branch_trigger_steps_gracefully_reduces_when_window_is_short():
    rng = np.random.default_rng(1)
    trigger_steps = anchored.sample_branch_trigger_steps(
        rng=rng,
        branch_min_step=10,
        branch_max_step=14,
        branches_per_rollout=4,
        min_gap=5,
    )

    assert len(trigger_steps) == 1
    assert 10 <= trigger_steps[0] <= 14


def test_compute_saved_branch_length_matches_spec():
    assert (
        anchored.compute_saved_branch_length(perturb_steps=3, recovery_steps=15) == 17
    )
    assert anchored.compute_saved_branch_length(perturb_steps=0, recovery_steps=15) == 15
    assert anchored.compute_saved_branch_length(perturb_steps=1, recovery_steps=0) == 0


def test_validate_anchored_resume_attrs_rejects_mismatch():
    config = anchored.AnchoredRecoveryConfig(
        branches_per_rollout=3,
        branch_min_step=10,
        branch_max_step=25,
        perturb_steps=3,
        recovery_steps=15,
        noise_scale=1.0,
        arm_base_noise=0.1,
        arm_wrist_noise=0.1,
        thumb_noise=0.1,
        index_noise=0.1,
        middle_noise=0.1,
        ring_noise=0.1,
        pinky_noise=0.1,
        ou_theta=0.15,
        ou_mu=0.0,
        ou_dt=1.0,
    )
    root = _FakeRoot(
        1,
        attrs={
            "variant": "noisy_clean",
            "noise_strategy": "anchored_recovery",
            "episode_semantics": "anchored_branch_segment",
            "saved_segment_policy": "burst_plus_recovery",
            "executed_action_clipped": True,
            "anchored_branches_per_rollout": 3,
            "anchored_branch_min_step": 10,
            "anchored_branch_max_step": 24,
            "anchored_perturb_steps": 3,
            "anchored_recovery_steps": 15,
            "anchored_saved_branch_steps": 17,
            "noise_scale": 1.0,
            "ou_theta": 0.15,
            "ou_mu": 0.0,
            "ou_dt": 1.0,
            "arm_base_noise": 0.1,
            "arm_wrist_noise": 0.1,
            "thumb_noise": 0.1,
            "index_noise": 0.1,
            "middle_noise": 0.1,
            "ring_noise": 0.1,
            "pinky_noise": 0.1,
        },
    )

    with pytest.raises(ValueError, match="Anchored recovery resume attrs mismatch"):
        anchored.validate_anchored_resume_attrs(
            root,
            variant="noisy_clean",
            config=config,
        )


def test_rl_player_state_round_trip_clones_nested_tensors():
    class _StubPlayer:
        def __init__(self):
            self.states = (
                torch.tensor([[1.0, 2.0]]),
                [torch.tensor([[3.0, 4.0]])],
            )

    player = rl_player_module.RlPlayer.__new__(rl_player_module.RlPlayer)
    player.player = _StubPlayer()

    state = player.get_internal_state()
    assert state[0] is not player.player.states[0]
    assert state[1][0] is not player.player.states[1][0]

    state[0].add_(10.0)
    assert torch.equal(player.player.states[0], torch.tensor([[1.0, 2.0]]))

    player.set_internal_state(state)
    assert torch.equal(player.player.states[0], torch.tensor([[11.0, 12.0]]))
    state[0].add_(10.0)
    assert torch.equal(player.player.states[0], torch.tensor([[11.0, 12.0]]))


def test_stage5_parse_args_rejects_anchored_without_noisy_clean(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage5_collect_dataset.py",
            "--collection-type",
            "pickup",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "clean",
        ],
    )
    with pytest.raises(ValueError, match="requires --variant noisy_clean"):
        stage5_collector.parse_args()


def test_stage5_parse_args_rejects_anchored_multi_env(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage5_collect_dataset.py",
            "--collection-type",
            "pickup",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "noisy_clean",
            "--num-envs",
            "2",
        ],
    )
    with pytest.raises(ValueError, match="requires --num-envs=1"):
        stage5_collector.parse_args()


def test_stage5_parse_args_rejects_anchored_noisy_worker(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage5_collect_dataset.py",
            "--collection-type",
            "pickup",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "noisy_clean",
            "--num-envs",
            "1",
            "--noisy-worker",
        ],
    )
    with pytest.raises(ValueError, match="does not support --noisy-worker"):
        stage5_collector.parse_args()


def test_stage6_parse_args_rejects_anchored_multi_env(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dataset_pick_place_release.py",
            "--collection-type",
            "pick_place",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "noisy_clean",
            "--num-envs",
            "2",
        ],
    )
    with pytest.raises(ValueError, match="requires --num-envs=1"):
        stage6_collector.parse_args()


def test_stage6_parse_args_rejects_anchored_without_noisy_clean(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dataset_pick_place_release.py",
            "--collection-type",
            "pick_place",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "clean",
        ],
    )
    with pytest.raises(ValueError, match="requires --variant noisy_clean"):
        stage6_collector.parse_args()


def test_stage6_parse_args_rejects_anchored_noisy_worker(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dataset_pick_place_release.py",
            "--collection-type",
            "pick_place",
            "--noise-strategy",
            "anchored_recovery",
            "--variant",
            "noisy_clean",
            "--num-envs",
            "1",
            "--noisy-worker",
        ],
    )
    with pytest.raises(ValueError, match="does not support --noisy-worker"):
        stage6_collector.parse_args()
