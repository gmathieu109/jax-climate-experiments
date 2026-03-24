#!/usr/bin/env python3
"""
Chunked JCM simulation using ERA5 data.

Usage:
    python run_chunked.py  # defaults: 2yr, 90-day chunks
    python run_chunked.py --total_days 1825 --chunk_days 90
    python run_chunked.py --output my_run.nc
"""

import argparse
import numpy as np
import jax.numpy as jnp
import xarray as xr
import pandas as pd
from pathlib import Path
from importlib import resources

from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import PhysicsState
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters


def build_initial_state(era5_ds, coords, init_time):
    """Interpolate ERA5 onto JCM grid and build a PhysicsState."""
    MODEL_SIGMA  = coords.vertical.centers
    MODEL_P_HPA  = np.array(MODEL_SIGMA) * 1013.25
    MODEL_LATS_DEG = coords.horizontal.latitudes  * 180 / np.pi
    MODEL_LONS_DEG = coords.horizontal.longitudes * 180 / np.pi

    VARS_3D = ["temperature", "specific_humidity", "u_wind", "v_wind", "geopotential"]
    VARS_2D = ["surface_pressure"]

    ds = era5_ds.sel(time=init_time)

    era5_log_p   = np.log(ds.level.values.astype(np.float64))
    target_log_p = np.log(MODEL_P_HPA)

    ds_3d = ds[VARS_3D].assign_coords(level=era5_log_p).interp(
        {"level": target_log_p, "lat": MODEL_LATS_DEG, "lon": MODEL_LONS_DEG},
        kwargs={"fill_value": "extrapolate"},
    ).assign_coords(level=MODEL_SIGMA)

    ds_2d = ds[VARS_2D].interp(
        {"lat": MODEL_LATS_DEG, "lon": MODEL_LONS_DEG},
        kwargs={"fill_value": "extrapolate"},
    )
    ds_interp = xr.merge([ds_3d, ds_2d])

    return PhysicsState(
        u_wind=jnp.asarray(ds_interp["u_wind"].values),
        v_wind=jnp.asarray(ds_interp["v_wind"].values),
        temperature=jnp.asarray(ds_interp["temperature"].values),
        specific_humidity=jnp.asarray(ds_interp["specific_humidity"].values) * 1000.0,
        geopotential=jnp.asarray(ds_interp["geopotential"].values),
        normalized_surface_pressure=jnp.asarray(ds_interp["surface_pressure"].values) / 1e5,
    )


def main():
    parser = argparse.ArgumentParser(description="Chunked JCM simulation")
    parser.add_argument("--total_days", type=int, default=2*365,
                        help="Total simulation length in days (default: 730)")
    parser.add_argument("--chunk_days", type=int, default=90,
                        help="Days per chunk (default: 90)")
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save interval in days (default: 1.0)")
    parser.add_argument("--output", default="jcm_full_run.nc",
                        help="Output netCDF path (default: jcm_full_run.nc)")
    parser.add_argument("--init_time", default="2000-01-01T00:00:00",
                        help="ERA5 initialisation time (default: 2000-01-01)") # I would not cvhange this yet, seems hardcoded in some parts of the code
    parser.add_argument(
        "--outdir",
        default=None,
        help="Experiment output directory. Final .nc will be saved in outdir/run_nc/",
    )
    args = parser.parse_args()

    total_days    = args.total_days
    chunk_days    = args.chunk_days
    save_interval = args.save_interval
    if args.outdir is not None:
        base_outdir = Path(args.outdir)
        run_nc_dir = base_outdir / "run_nc"
        run_nc_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_nc_dir / Path(args.output).name
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_path.parent / f"{out_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────
    coords   = get_speedy_coords()
    data_dir = resources.files("jcm.data.bc.t30.clim")

    realistic_terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)
    realistic_forcing = ForcingData.from_file(data_dir / "forcing.nc", coords=coords)

    params = Parameters.default()
    model  = Model(
        coords=coords,
        terrain=realistic_terrain,
        physics=SpeedyPhysics(parameters=params),
    )

    # ── ERA5 init ──────────────────────────────────────────────────────
    init_time = pd.Timestamp(args.init_time)
    print(f"Opening ERA5 for initialisation at {init_time} …")
    era5_ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
        consolidated=True,
        storage_options={"token": "anon"},
    ).rename({
        "latitude": "lat", "longitude": "lon",
        "u_component_of_wind": "u_wind",
        "v_component_of_wind": "v_wind",
    })

    init_state = build_initial_state(era5_ds, coords, init_time)
    era5_ds.close()
    print("  Initial state built.\n")

    # ── Chunked integration ────────────────────────────────────────────
    n_chunks    = int(np.ceil(total_days / chunk_days))
    chunk_paths = []

    # First chunk uses model.run (sets the initial state)
    # Subsequent chunks use model.resume (continues from last state)
    remaining = total_days

    for i in range(n_chunks):
        this_chunk = min(chunk_days, remaining)
        chunk_file = chunk_dir / f"chunk_{i:03d}.nc"

        print(f"── Chunk {i+1}/{n_chunks}: {this_chunk} days ──")

        if i == 0:
            predictions = model.run(
                initial_state=init_state,
                forcing=realistic_forcing,
                total_time=this_chunk,
                save_interval=save_interval,
            )
        else:
            predictions = model.resume(
                forcing=realistic_forcing,
                total_time=this_chunk,
                save_interval=save_interval,
            )

        ds_chunk = predictions.to_xarray(physics_module=model.physics)
        ds_chunk.to_netcdf(chunk_file)
        chunk_paths.append(chunk_file)
        print(f"   Saved → {chunk_file}  "
              f"(T range: {float(ds_chunk['temperature'].min()):.1f}–"
              f"{float(ds_chunk['temperature'].max()):.1f} K, "
              f"NaN: {int(ds_chunk['temperature'].isnull().sum())})\n")

        # Free GPU memory held by predictions
        del predictions, ds_chunk

        remaining -= this_chunk

    # ── Concatenate ────────────────────────────────────────────────────
    print(f"Concatenating {len(chunk_paths)} chunks → {out_path}")
    datasets = [xr.open_dataset(p) for p in chunk_paths]
    ds_full  = xr.concat(datasets, dim="time")
    ds_full.to_netcdf(out_path)
    for d in datasets:
        d.close()

    print(f"Done. Output: {out_path}")
    print(f"  Time: {ds_full.time.values[0]} → {ds_full.time.values[-1]}")
    print(f"  Shape: {dict(ds_full.sizes)}")
    print(f"  T range: {float(ds_full['temperature'].min()):.1f}–{float(ds_full['temperature'].max()):.1f} K")


if __name__ == "__main__":
    main()
