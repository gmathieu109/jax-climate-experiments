#!/usr/bin/env ~/extras/.conda/envs/jcm/bin/python

import xarray as xr
import numpy as np

forcing_file = "/home/gmathieu/code_uqam/src/jax-gcm/jcm/data/bc/t30/clim/forcing.nc"

ds = xr.open_dataset(forcing_file)

print("\n===== VARIABLES =====")
for v in ds.data_vars:
    print(f"{v:10s} {ds[v].dims} {ds[v].shape}")

print("\n===== COORDS =====")
print(ds.coords)

print("\n===== STATS =====")
for v in ds.data_vars:
    da = ds[v]
    print(f"\n--- {v} ---")
    print("min:", float(da.min()))
    print("max:", float(da.max()))
    print("mean:", float(da.mean()))

print("\n===== TIME =====")
if "time" in ds:
    print(ds.time.values[:10])
    print("n_time =", ds.dims["time"])