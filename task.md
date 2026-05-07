Stage 1: Single-env viewer rollout with the canonical DexToolBench task
Goal: confirm the expert + env + policy loading pipeline works end-to-end with a known-good configuration before introducing any of my changes.
    •    num_envs=1, viewer enabled.
    •    Object: claw_hammer.
    •    Goals: load claw_hammer/swing_down.json directly, use those goals unchanged.
    •    Horizon: 150 steps.
    •    No camera capture yet, no data saving yet — just watch the viewer.
Success criterion: I can visually confirm the expert reaches down, grasps the hammer, and lifts/swings it. If this fails, nothing else matters.
Please write this as a minimal Python script I can run.

Stage 2: Replace task goals with a custom lifted goal
Goal: verify the expert behaves correctly when given a custom single-goal pickup target instead of the full DexToolBench trajectory.
    •    Same setup as Stage 1.
    •    Replace traj_data["goals"] with [start_pose + [0, 0, 0.15, 0, 0, 0, 0]] (15cm lift, same orientation).
    •    Run 3 variants in sequence: 0.15m, 0.20m, and "first 3 goals of swing_down" (option 3 from your earlier message).
Success criterion: at least one of these three configurations produces a clean pickup. Note which works best — that's our V1 goal config.

Stage 3: Add scene camera capture
Goal: verify camera rendering is synchronized with policy steps and produces useful images.
    •    Use the best goal config from Stage 2.
    •    Add scene camera with the framing we discussed (target between hand at y=0.8 and tool at y=0).
    •    Render at every step using the order: render image → query policy → save tuple → step env.
    •    Save 1 rollout's worth of (image_t, obs_t, action_t) tuples.
    •    After the rollout, generate a GIF of the rendered images so I can visually inspect the trajectory.
Success criterion: the GIF shows the hand approaching, grasping, and lifting the hammer. The tool and the gripper region are clearly visible throughout.

Stage 4: Add random tool start positions
Goal: introduce diversity in start poses so the dataset isn't all from one fixed initial condition, but in a way that keeps the goal pose synchronized with the start pose.
    •    Use the approach you suggested: re-create the env per rollout (or per small batch), compute lifted_pose = randomized_start + [0, 0, 0.20] for each rollout.
    •    Random x/y range: e.g. ±10cm around nominal start, no rotation jitter for V1.
    •    Run 10 rollouts with different randomized starts.
Success criterion: ≥7/10 successful pickups (matching your "60–80% on mild random x/y" estimate). If success rate is below this, debug before scaling up.

Stage 5: Parallelize and collect the actual dataset
Goal: scale up to many envs and many rollouts to produce a usable dataset.
    •    Bump num_envs to as many as memory allows with rendering enabled (start with 16, see what fits).
    •    Run enough rollouts to hit ~50K transitions (this may take several hours).
    •    Save in the Diffusion Policy-compatible format (Zarr with meta/episode_ends, data/img, data/state, data/action).
    •    Only the clean dataset for now — no PDP state perturbation yet.
Success criterion: dataset on disk, loadable via Diffusion Policy's dataloader, contains episodes with successful pickups.

Stage 6 (later): Add PDP-style noise variants
Goal: generate the noisy-clean and noisy-noisy variants for the methodological ablation. We'll tackle this after Stage 5 is verified working.
