# Data Collection and Processing

This document describes the full pipeline for collecting real-world object manipulation data and converting it into task trajectories for DexToolBench.

## FoundationPose Setup

Clone and set up the [FoundationPose fork](https://github.com/kushal2000/FoundationPose) in a **separate conda environment** (`foundationpose`). See its README for full installation instructions, including model weight downloads and ROS setup.

This environment is used for Steps 1, 3, and live inference.

## Step 1: Record RGB-D Video

From the FoundationPose repo, record an RGB-D video with a ZED camera:

```bash
conda activate foundationpose

python record_video.py \
    --save_dir recordings/ \
    --serial_number <ZED_SERIAL> \
    --fps 30
```

Press `Ctrl+C` to stop. This creates:

```
recordings/<timestamp>/
  ├── rgb/frame_0000.png, frame_0001.png, ...
  ├── depth/frame_0000.png, frame_0001.png, ...
  ├── cam_K.txt
  └── rgb.mp4
```

## Step 2: Extract Object Mesh (SAM 2 + SAM 3D)

> **TODO**: This section will be filled in by Tyler.

Link: TBD

### Installation

TBD

### Usage

TBD

## Step 3: Extract 6D Poses with FoundationPose

Run FoundationPose on the recorded video:

```bash
conda activate foundationpose

python extract_poses.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --calibration /path/to/T_RC.txt \
    --output_path recordings/<timestamp>/poses.json \
    --debug 1
```

An interactive window opens on the first frame -- click 4 corners of a bounding box around the object for SAM segmentation. FoundationPose then tracks through all frames.

Output (`poses.json`):
```json
{
  "poses_cam": [[x, y, z, qx, qy, qz, qw], ...],
  "poses_robot": [[x, y, z, qx, qy, qz, qw], ...]
}
```

## Step 4: Process Poses into Task Trajectory

From the SimToolReal repo, process the raw poses into a DexToolBench task trajectory:

```bash
conda activate simtoolreal

python dextoolbench/process_poses.py \
    --poses_path recordings/<timestamp>/poses.json \
    --object_category hammer \
    --object_name claw_hammer \
    --task_name swing_down
```

This outputs a trajectory JSON to `dextoolbench/trajectories/<object_category>/<object_name>/<task_name>.json` with poses in world frame, ready for use in training and evaluation.

## FoundationPose During Inference

To run live tracking at inference time, install FoundationPose as described above, then run:

```bash
conda activate foundationpose
cd /path/to/FoundationPose

python live_tracking_with_ros.py \
    --mesh_path /path/to/object.obj \
    --calibration calibration/T_RC_example.txt
```

This publishes object poses to `robot_frame/current_object_pose` as a ROS `PoseStamped` topic, which is consumed by the RL Policy Node. See the main [README](../README.md) for the full deployment flowchart.
