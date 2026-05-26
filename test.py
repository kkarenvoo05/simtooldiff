import zarr

clean = zarr.open("/move/u/chrzhang/simtooldiff/data/stage5_train.zarr", mode="r")
noisy = zarr.open("data/pickup_train_4x_out0.15.zarr", mode="r")

print("=== CLEAN ===")
print("transitions:", clean["data"]["img"].shape[0])
print("episodes:", clean["meta"]["episode_ends"].shape[0])
print("attrs:", dict(clean.attrs))

print("\n=== NOISY 4× ===")
print("transitions:", noisy["data"]["img"].shape[0])
print("episodes:", noisy["meta"]["episode_ends"].shape[0])
print("attrs:", dict(noisy.attrs))