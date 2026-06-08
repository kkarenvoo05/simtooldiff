"""Pure unit tests for stage6_noise_consistency_probe.

No Isaac Gym, no policy, no env — only functions that operate on numpy arrays
and plain Python.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out isaacgym so the module can be imported without a GPU/gym install.
# ---------------------------------------------------------------------------

isaacgym_stub = types.ModuleType("isaacgym")
isaacgym_stub.gymapi = object()
sys.modules.setdefault("isaacgym", isaacgym_stub)

# Stub heavy runtime deps that are imported at module level.
for _mod in (
    "torch",
    "omegaconf",
    "zarr",
    "numcodecs",
    "imageio",
    "imageio.v2",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "matplotlib",
    "matplotlib.pyplot",
):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# torch stub: just enough for the module to not crash at import.
_torch = sys.modules["torch"]
_torch.zeros = lambda *a, **kw: np.zeros(a)  # type: ignore[assignment]
_torch.tensor = lambda *a, **kw: np.array(a[0])  # type: ignore[assignment]
_torch.manual_seed = lambda s: None
_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
_torch.random = types.SimpleNamespace(fork_rng=lambda **kw: None)
_torch.clamp = lambda t, lo, hi: np.clip(t, lo, hi)
_torch.arange = lambda *a, **kw: np.arange(*a)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub stage5_noise_config before importing our module.
_noise_config_stub = types.ModuleType("stage5_noise_config")
_noise_config_stub.BASE_NOISE_DEFAULTS = {  # type: ignore[attr-defined]
    "arm_base": 0.005, "arm_wrist": 0.01, "thumb": 0.015,
    "index": 0.01, "middle": 0.01, "ring": 0.008, "pinky": 0.008,
}
def _resolve_noise_config_stub(*, noise_scale, arm_base_noise, arm_wrist_noise,
                                thumb_noise, index_noise, middle_noise, ring_noise, pinky_noise):
    base = _noise_config_stub.BASE_NOISE_DEFAULTS
    scaled = {k: v * noise_scale for k, v in base.items()}
    overrides = dict(arm_base=arm_base_noise, arm_wrist=arm_wrist_noise, thumb=thumb_noise,
                     index=index_noise, middle=middle_noise, ring=ring_noise, pinky=pinky_noise)
    for k, v in overrides.items():
        if v is not None:
            scaled[k] = v
    return scaled
_noise_config_stub.resolve_noise_config = _resolve_noise_config_stub  # type: ignore[attr-defined]
sys.modules["stage5_noise_config"] = _noise_config_stub

import stage6_noise_consistency_probe as probe  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: synthetic rollout record
# ---------------------------------------------------------------------------

def _make_record(
    ee_positions: np.ndarray,
    clean_actions: np.ndarray,
    success: bool = True,
    gate_history: Optional[List[bool]] = None,
    object_poses: Optional[np.ndarray] = None,
):
    from stage6_noise_consistency_probe import RolloutRecord
    T = ee_positions.shape[0]
    if gate_history is None:
        gate_history = [False] * T
    if object_poses is None:
        object_poses = np.zeros((T, 7), dtype=np.float32)
    return RolloutRecord(
        rollout_type="noisy",
        rollout_idx=0,
        init_idx=0,
        success=success,
        steps=T,
        ee_positions=ee_positions.astype(np.float32),
        object_poses=object_poses.astype(np.float32),
        clean_actions=clean_actions.astype(np.float32),
        exec_actions=clean_actions.astype(np.float32),
        pickup_gate_history=gate_history,
        frames=None,
    )


# ---------------------------------------------------------------------------
# Test: deterministic noise seed
# ---------------------------------------------------------------------------

class TestNoiseSeed:
    def test_same_inputs_same_seed(self):
        assert probe._noise_seed(42, 0, 0) == probe._noise_seed(42, 0, 0)
        assert probe._noise_seed(0, 3, 7) == probe._noise_seed(0, 3, 7)

    def test_different_rollout_idx_different_seed(self):
        s0 = probe._noise_seed(42, 0, 0)
        s1 = probe._noise_seed(42, 0, 1)
        s2 = probe._noise_seed(42, 0, 2)
        assert len({s0, s1, s2}) == 3, "All three seeds should be distinct"

    def test_different_init_idx_different_seed(self):
        s0 = probe._noise_seed(0, 0, 0)
        s1 = probe._noise_seed(0, 1, 0)
        assert s0 != s1

    def test_different_base_seed_different_seed(self):
        assert probe._noise_seed(0, 0, 0) != probe._noise_seed(1, 0, 0)

    def test_formula_correctness(self):
        # seed + init_idx * 1000 + rollout_idx
        assert probe._noise_seed(7, 3, 5) == 7 + 3 * 1000 + 5


# ---------------------------------------------------------------------------
# Test: metric 1 — EE deviation from reference
# ---------------------------------------------------------------------------

class TestMetric1EEDeviation:
    def test_identical_trajectories_give_zero_deviation(self):
        ref = np.zeros((50, 3), dtype=np.float32)
        noisy_list = [np.zeros((50, 3), dtype=np.float32) for _ in range(5)]
        mean_dev, std_dev = probe._metric1_ee_dev_from_ref(ref, noisy_list)
        np.testing.assert_allclose(mean_dev, 0.0, atol=1e-6)
        np.testing.assert_allclose(std_dev, 0.0, atol=1e-6)

    def test_known_constant_offset(self):
        T = 20
        ref = np.zeros((T, 3), dtype=np.float32)
        # All noisy rollouts are offset by 1.0 in x → distance is always 1.0.
        noisy_list = [np.column_stack([np.ones(T), np.zeros(T), np.zeros(T)]).astype(np.float32)]
        mean_dev, std_dev = probe._metric1_ee_dev_from_ref(ref, noisy_list)
        np.testing.assert_allclose(mean_dev, 1.0, atol=1e-5)
        np.testing.assert_allclose(std_dev, 0.0, atol=1e-5)

    def test_mixed_offsets_mean(self):
        T = 10
        ref = np.zeros((T, 3), dtype=np.float32)
        # Two rollouts: distances 1 and 3 → mean 2.
        n1 = np.column_stack([np.ones(T), np.zeros(T), np.zeros(T)]).astype(np.float32)
        n2 = np.column_stack([np.full(T, 3.0), np.zeros(T), np.zeros(T)]).astype(np.float32)
        mean_dev, _ = probe._metric1_ee_dev_from_ref(ref, [n1, n2])
        np.testing.assert_allclose(mean_dev, 2.0, atol=1e-5)

    def test_empty_noisy_list_returns_zeros(self):
        ref = np.ones((10, 3), dtype=np.float32)
        mean_dev, std_dev = probe._metric1_ee_dev_from_ref(ref, [])
        assert mean_dev.shape == (10,)
        np.testing.assert_allclose(mean_dev, 0.0)

    def test_length_alignment(self):
        ref = np.zeros((30, 3), dtype=np.float32)
        noisy_list = [np.ones((20, 3), dtype=np.float32)]
        mean_dev, _ = probe._metric1_ee_dev_from_ref(ref, noisy_list)
        assert mean_dev.shape == (20,)  # truncated to shorter


# ---------------------------------------------------------------------------
# Test: metric 2 — EE pairwise spread
# ---------------------------------------------------------------------------

class TestMetric2EESpread:
    def test_identical_rollouts_zero_spread(self):
        traj = np.zeros((30, 3), dtype=np.float32)
        noisy_list = [traj.copy() for _ in range(5)]
        std_t, max_t = probe._metric2_ee_pairwise_spread(noisy_list)
        np.testing.assert_allclose(std_t, 0.0, atol=1e-6)
        np.testing.assert_allclose(max_t, 0.0, atol=1e-6)

    def test_known_fan_max_pairwise(self):
        T = 5
        # Two rollouts 2m apart in x.
        r1 = np.zeros((T, 3), dtype=np.float32)
        r2 = np.column_stack([np.full(T, 2.0), np.zeros(T), np.zeros(T)]).astype(np.float32)
        _, max_t = probe._metric2_ee_pairwise_spread([r1, r2])
        np.testing.assert_allclose(max_t, 2.0, atol=1e-5)

    def test_single_rollout_returns_zeros(self):
        traj = np.ones((15, 3), dtype=np.float32)
        std_t, max_t = probe._metric2_ee_pairwise_spread([traj])
        np.testing.assert_allclose(std_t, 0.0)
        np.testing.assert_allclose(max_t, 0.0)


# ---------------------------------------------------------------------------
# Test: metric 3 — final object pose spread
# ---------------------------------------------------------------------------

class TestMetric3FinalObjectSpread:
    def test_identical_finals_zero_std(self):
        T = 5
        # All rollouts have the same object pose → spread should be zero.
        obj_pose = np.tile([0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0], (T, 1)).astype(np.float32)
        rollouts = [
            _make_record(np.zeros((T, 3)), np.zeros((T, 29)), object_poses=obj_pose)
            for _ in range(4)
        ]
        std = probe._metric3_final_obj_pos_spread(rollouts)
        np.testing.assert_allclose(std, 0.0, atol=1e-6)

    def test_known_spread(self):
        T = 5
        # Three rollouts with object x-positions at 0, 1, 2 → x-std ≈ 0.8165.
        poses = []
        for i in range(3):
            obj_poses = np.zeros((T, 7), dtype=np.float32)
            obj_poses[:, 0] = float(i)  # set x position
            poses.append(
                _make_record(
                    np.zeros((T, 3)),
                    np.zeros((T, 29)),
                    object_poses=obj_poses,
                )
            )
        std = probe._metric3_final_obj_pos_spread(poses)
        expected_x_std = np.std([0.0, 1.0, 2.0])
        assert abs(std[0] - expected_x_std) < 1e-5

    def test_empty_returns_zeros(self):
        std = probe._metric3_final_obj_pos_spread([])
        np.testing.assert_allclose(std, 0.0)


# ---------------------------------------------------------------------------
# Test: metric 4 — action-sequence consistency
# ---------------------------------------------------------------------------

class TestMetric4ActionConsistency:
    def test_identical_sequences_zero_distance(self):
        T, D = 20, 29
        actions = np.ones((T, D), dtype=np.float32)
        rollouts = [_make_record(np.zeros((T, 3)), actions) for _ in range(4)]
        dist = probe._metric4_action_seq_consistency(rollouts)
        assert abs(dist) < 1e-6

    def test_single_rollout_zero_distance(self):
        T, D = 10, 29
        actions = np.random.rand(T, D).astype(np.float32)
        rollouts = [_make_record(np.zeros((T, 3)), actions)]
        dist = probe._metric4_action_seq_consistency(rollouts)
        assert dist == 0.0

    def test_orthogonal_sequences_nonzero(self):
        T, D = 5, 29
        a1 = np.zeros((T, D), dtype=np.float32)
        a2 = np.ones((T, D), dtype=np.float32)
        r1 = _make_record(np.zeros((T, 3)), a1)
        r2 = _make_record(np.zeros((T, 3)), a2)
        dist = probe._metric4_action_seq_consistency([r1, r2])
        expected = float(np.linalg.norm((a1 - a2).reshape(-1)))
        assert abs(dist - expected) < 1e-4

    def test_symmetry(self):
        T, D = 8, 29
        a1 = np.random.rand(T, D).astype(np.float32)
        a2 = np.random.rand(T, D).astype(np.float32)
        r1 = _make_record(np.zeros((T, 3)), a1)
        r2 = _make_record(np.zeros((T, 3)), a2)
        d12 = probe._metric4_action_seq_consistency([r1, r2])
        d21 = probe._metric4_action_seq_consistency([r2, r1])
        assert abs(d12 - d21) < 1e-6


# ---------------------------------------------------------------------------
# Test: band_verdict
# ---------------------------------------------------------------------------

class TestBandVerdict:
    def test_tube_when_always_below_threshold(self):
        T = 100
        dev = np.full(T, 0.01, dtype=np.float64)  # well below 0.03
        verdict = probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.10)
        assert verdict == "tube"

    def test_fan_when_high_from_midpoint(self):
        T = 100
        # Low early, then exceeds 0.10 from the midpoint.
        dev = np.concatenate([np.full(50, 0.01), np.full(50, 0.15)]).astype(np.float64)
        verdict = probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.10)
        assert verdict == "fan"

    def test_intermediate(self):
        T = 100
        # Exceeds tube threshold but stays below fan threshold throughout.
        dev = np.full(T, 0.06, dtype=np.float64)
        verdict = probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.10)
        assert verdict == "intermediate"

    def test_empty_array_returns_intermediate(self):
        verdict = probe._band_verdict(np.array([]), tube_threshold=0.03, fan_threshold=0.10)
        assert verdict == "intermediate"

    def test_tube_threshold_configurable(self):
        T = 50
        dev = np.full(T, 0.04, dtype=np.float64)
        # With tight threshold: 0.04 > 0.03 → not tube.
        assert probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.10) == "intermediate"
        # With generous threshold: 0.04 < 0.05 → tube.
        assert probe._band_verdict(dev, tube_threshold=0.05, fan_threshold=0.10) == "tube"

    def test_fan_threshold_configurable(self):
        T = 60
        dev = np.concatenate([np.full(30, 0.01), np.full(30, 0.08)]).astype(np.float64)
        # Default fan_threshold=0.10: 0.08 < 0.10 → intermediate.
        assert probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.10) == "intermediate"
        # With fan_threshold=0.07: 0.08 > 0.07 → fan.
        assert probe._band_verdict(dev, tube_threshold=0.03, fan_threshold=0.07) == "fan"


# ---------------------------------------------------------------------------
# Test: CLI validation — --num-envs != 1 is rejected
# ---------------------------------------------------------------------------

class TestCLIValidation:
    def test_num_envs_not_1_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["stage6_noise_consistency_probe.py", "--num-envs", "4"],
        )
        with pytest.raises(SystemExit) as exc_info:
            probe.parse_args()
        # argparse calls parser.error() which calls sys.exit(2).
        assert exc_info.value.code == 2

    def test_num_envs_1_is_accepted(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["stage6_noise_consistency_probe.py", "--num-envs", "1"],
        )
        args = probe.parse_args()
        assert args.num_envs == 1

    def test_default_num_envs_is_1(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["stage6_noise_consistency_probe.py"])
        args = probe.parse_args()
        assert args.num_envs == 1


# ---------------------------------------------------------------------------
# Test: _first_pickup_hold_step
# ---------------------------------------------------------------------------

class TestPickupHoldStep:
    def test_no_gate_returns_none(self):
        assert probe._first_pickup_hold_step([False] * 20, 5) is None

    def test_finds_first_hold(self):
        history = [False] * 5 + [True] * 5 + [False] * 5
        result = probe._first_pickup_hold_step(history, 3)
        # Consecutive hits 3 at index 7 (0-indexed: 5,6,7).
        assert result == 7

    def test_hold_steps_1(self):
        history = [False, False, True, False]
        result = probe._first_pickup_hold_step(history, 1)
        assert result == 2


# ---------------------------------------------------------------------------
# Test: metric5 pickup timing spread
# ---------------------------------------------------------------------------

class TestMetric5PickupTiming:
    def test_identical_timing_zero_std(self):
        T = 30
        # Each rollout: pickup gate triggers at step 10 and holds ≥ HOLD_STEPS.
        gate = [False] * 10 + [True] * 20
        rollouts = [
            _make_record(np.zeros((T, 3)), np.zeros((T, 29)), gate_history=gate)
            for _ in range(4)
        ]
        std, values = probe._metric5_pickup_timing_spread(rollouts)
        assert all(v is not None for v in values)
        assert std == 0.0

    def test_spread_matches_numpy_std(self):
        T = 50
        steps = [10, 15, 20, 25]  # pickup hold at these steps (adjusted for hold_steps=5)
        rollouts = []
        for s in steps:
            gate = [False] * (s - 4) + [True] * (T - (s - 4))
            rollouts.append(
                _make_record(np.zeros((T, 3)), np.zeros((T, 29)), gate_history=gate)
            )
        std, values = probe._metric5_pickup_timing_spread(rollouts)
        numeric = [v for v in values if v is not None]
        expected_std = float(np.std(numeric))
        assert abs(std - expected_std) < 1e-5


# ---------------------------------------------------------------------------
# Test: summary dict structure
# ---------------------------------------------------------------------------

class TestSummaryDict:
    def _make_init_metrics(self, init_idx: int, verdict: str) -> probe.InitMetrics:
        return probe.InitMetrics(
            init_idx=init_idx,
            start_pose=[0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0],
            reset_fidelity_passed=True,
            reset_fidelity_max_err=1e-6,
            ee_dev_mean=[0.01] * 50,
            ee_dev_std=[0.001] * 50,
            ee_spread_mean=[0.005] * 50,
            ee_spread_max=[0.01] * 50,
            final_obj_pos_std=[0.01, 0.01, 0.01],
            action_seq_mean_pairwise_l2=0.5,
            pickup_timing_std=2.0,
            pickup_timing_values=[20, 22, 18, 19],
            noisy_success_rate=0.8,
            noisy_successes=8,
            rollouts_per_init=10,
            band_verdict=verdict,
        )

    def _make_fake_args(self):
        import argparse
        args = argparse.Namespace(
            noise_scale=10.0,
            arm_base_noise=0.05, arm_wrist_noise=0.1, thumb_noise=0.15,
            index_noise=0.1, middle_noise=0.1, ring_noise=0.08, pinky_noise=0.08,
            ou_theta=0.15, ou_mu=0.0, ou_dt=1.0,
            object_category="hammer", object_name="claw_hammer",
            task_name="swing_down", num_initializations=3,
            rollouts_per_init=10, horizon=250, seed=0,
            tube_threshold=0.03, fan_threshold=0.10,
        )
        return args

    def test_summary_has_required_top_level_keys(self):
        metrics = [self._make_init_metrics(i, "tube") for i in range(3)]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=120.0)
        for key in ("band_verdict", "band_verdict_per_init", "noise_config",
                    "run_config", "aggregate", "per_init", "elapsed_s"):
            assert key in summary, f"Missing key: {key}"

    def test_majority_tube_verdict(self):
        metrics = [
            self._make_init_metrics(0, "tube"),
            self._make_init_metrics(1, "tube"),
            self._make_init_metrics(2, "fan"),
        ]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=60.0)
        assert summary["band_verdict"] == "tube"

    def test_majority_fan_verdict(self):
        metrics = [
            self._make_init_metrics(0, "fan"),
            self._make_init_metrics(1, "fan"),
            self._make_init_metrics(2, "tube"),
        ]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=60.0)
        assert summary["band_verdict"] == "fan"

    def test_split_verdict_gives_intermediate(self):
        metrics = [
            self._make_init_metrics(0, "tube"),
            self._make_init_metrics(1, "fan"),
        ]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=60.0)
        assert summary["band_verdict"] == "intermediate"

    def test_per_init_length_matches(self):
        N = 4
        metrics = [self._make_init_metrics(i, "intermediate") for i in range(N)]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=10.0)
        assert len(summary["per_init"]) == N

    def test_reset_fidelity_fields_present(self):
        metrics = [self._make_init_metrics(0, "tube")]
        summary = probe._summary_dict(metrics, self._make_fake_args(), elapsed_s=5.0)
        per = summary["per_init"][0]
        assert "reset_fidelity" in per
        assert per["reset_fidelity"]["passed"] is True

    def test_noise_config_in_summary(self):
        metrics = [self._make_init_metrics(0, "tube")]
        args = self._make_fake_args()
        summary = probe._summary_dict(metrics, args, elapsed_s=1.0)
        nc = summary["noise_config"]
        assert nc["noise_scale"] == 10.0
        assert nc["ou_theta"] == 0.15
