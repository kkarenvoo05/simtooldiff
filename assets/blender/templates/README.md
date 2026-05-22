# SimToolDiff Blender Template Contract

`simtool_lab.blend` is the master static-scene template for closed-loop
photorealistic policy evaluation.

## What The Template Owns

You may edit these in Blender and save the `.blend`:

- World/HDRI environment. The HDRI should stay hidden from camera rays and
  should act as low-strength ambient fill/reflection only.
- Static area lights. The default template uses the `reference_area` preset:
  `OverheadStrip`, `CameraSoftbox`, `RearRim`, and `ShadowFill`.
- Static background furniture, walls, floor, rear bench, and non-physics props.
- Static scene materials for the table, floor, block, bench, walls, and lights.
- Procedural/static material node graphs for objects already in the template.

The intended lighting design is deliberately simple:

- `OverheadStrip`: rectangular area light above/front of the workcell. This
  gives the reference-style lab strip-light reflection and main downward shape.
- `CameraSoftbox`: front/side softbox that lights the arm face without making
  the whole scene ambient.
- `RearRim`: weak rear/side edge light for the hammer and arm silhouette.
- `ShadowFill`: very large, dim fill. It only lifts shadow cores.
- HDRI: strength around 0.02 to 0.05, hidden from camera. It should contribute
  subtle reflection color only, not become the visible background or dominant
  light source.

For scripted comparisons, set `SIMTOOLDIFF_LIGHTING_PRESET` before launching
Blender. Supported values are `reference_area`, `softbox_grid`, `spot_accent`,
and `clean_key_fill`.

## What Runtime Owns

The closed-loop evaluator overwrites or creates these every run:

- `EvalCamera`: position, rotation, focal length, sensor size, resolution.
- Render engine settings: Cycles/EEVEE choice, samples, deterministic seed,
  no motion blur, no depth of field, output resolution.
- Robot and hand meshes. They are imported from the URDF mesh manifest at
  startup and named by link, for example `iiwa14_link_1`.
- Tool mesh. It is imported dynamically from the task object name and named
  `tool_<object_name>`, for example `tool_claw_hammer`.
- Every robot link pose and tool pose on every frame. IsaacGym physics is the
  source of truth.

## Do Not Modify

Do not rename, delete, move, or rescale these contract objects unless you also
intend to change the evaluation visual contract:

- `table`: must stay centered at `(0, 0, 0.38)` with dimensions
  `(0.475, 0.4, 0.3)`.
- `table_nail`: must stay at `(-0.16, 0.06, 0.555)` with dimensions
  `(0.03, 0.03, 0.06)`.
- `floor`: must exist as the static floor/shadow receiver.
- `clean_back_wall`: the clean visible background. You may change its material,
  but keep a physical background object if the HDRI stays hidden from camera.

The renderer validates these names when `--blend-file` is used and warns if the
locations or dimensions drift from the IsaacGym visual contract.

Do not place robot or tool objects manually in this template. They are imported
and placed from IsaacGym state during evaluation.

## Rebuild

From the repository root:

```bash
/tmp/blender-4.2.9-linux-x64/blender --background \
  --python assets/blender/templates/build_simtool_lab_template.py
```

## Closed-Loop Usage

```bash
.venv/bin/python blender_eval/eval_blender.py \
  --worker \
  --checkpoint /home/takaraet/Projects/cs224r/checkpoints/epoch=0050-val_loss=0.0465.ckpt \
  --renderer blender \
  --engine cycles \
  --samples 64 \
  --cycles-device auto \
  --blend-file assets/blender/templates/simtool_lab.blend \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --num-envs 1 \
  --episodes-per-object 1 \
  --horizon 250 \
  --video-dir /home/takaraet/Projects/cs224r/blender_eval_videos/template_test \
  --result-json /tmp/template_test_result.json
```

For a full driver run over a split, pass the same `--blend-file` without
`--worker`; the driver passes it through to every object worker.

On an 8 GB GPU, use `--cycles-device cpu` if Blender exits or OOMs while the
policy and IsaacGym are also on CUDA. That keeps the closed loop correct and
only moves Cycles rendering off the GPU.
