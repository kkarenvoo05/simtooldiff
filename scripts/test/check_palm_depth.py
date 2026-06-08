import zarr, numpy as np, matplotlib.pyplot as plt

z = zarr.open("/move/u/karenvo/Projects/simtoolreal/data/pick_place_release_clean.zarr", "r")
ends = z["meta/episode_ends"][:]
starts = np.concatenate([[0], ends[:-1]])
state = z["data/state"]

# Sanity check: plot candidate palm x/y/z for episode 0
s, e = int(starts[0]), int(ends[0])
fig0, ax0 = plt.subplots()
for idx, name in [(87, "x?"), (88, "y?"), (89, "z?")]:
    ax0.plot(state[s:e, idx], label=f"dim {idx} ({name})")

ax0.legend(); ax0.set_title("Identify palm dims")
fig0.savefig("palm_dim_check.png")

TABLE_Z = 0.38
PALM_Z = 89  # state_list: joint_pos(29)+joint_vel(29)+prev_act(29)+palm_x+palm_y+palm_z

fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for ax, ep_idx in zip(axes.flat, range(9)):
    s, e = int(starts[ep_idx]), int(ends[ep_idx])
    palm_z = state[s:e, PALM_Z]
    ax.plot(palm_z, label="palm z")
    ax.axhline(TABLE_Z, color="r", linestyle="--", label="table z=0.38")
    ax.set_title(f"Episode {ep_idx}")
    ax.set_ylim(0.3, 0.9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("palm_z_episodes.png")