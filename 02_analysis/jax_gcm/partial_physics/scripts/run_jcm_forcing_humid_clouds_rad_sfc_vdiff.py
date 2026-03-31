#!/usr/bin/env python3
"""
Chunked JCM partial-physics simulation using ERA5 data.

Active terms:
- set_physics_flags
- set_forcing
- spec_hum_to_rel_hum
- get_clouds
- get_shortwave_rad_fluxes
- get_downward_longwave_rad_fluxes
- get_upward_longwave_rad_fluxes

Disabled terms:
- convection
- large-scale condensation
- surface fluxes
- vertical diffusion

Usage:
    ~/extras/.conda/envs/jcm/bin/python run_jcm_forcing_humid_clouds_rad.py
    ~/extras/.conda/envs/jcm/bin/python run_jcm_forcing_humid_clouds_rad.py --total_days 30 --chunk_days 10
"""

import argparse
from collections import abc
from pathlib import Path
from typing import Callable, Tuple
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.date import DateData
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_coords import get_speedy_coords, SpeedyCoords
from jcm.utils import tree_index_3d


def build_initial_state(era5_ds, coords, init_time):
    """Interpolate ERA5 onto JCM grid and build a PhysicsState."""
    model_sigma = coords.vertical.centers
    model_p_hpa = np.array(model_sigma) * 1013.25
    model_lats_deg = coords.horizontal.latitudes * 180 / np.pi
    model_lons_deg = coords.horizontal.longitudes * 180 / np.pi

    vars_3d = ["temperature", "specific_humidity", "u_wind", "v_wind", "geopotential"]
    vars_2d = ["surface_pressure"]

    ds = era5_ds.sel(time=init_time)

    era5_log_p = np.log(ds.level.values.astype(np.float64))
    target_log_p = np.log(model_p_hpa)

    ds_3d = ds[vars_3d].assign_coords(level=era5_log_p).interp(
        {"level": target_log_p, "lat": model_lats_deg, "lon": model_lons_deg},
        kwargs={"fill_value": "extrapolate"},
    ).assign_coords(level=model_sigma)

    ds_2d = ds[vars_2d].interp(
        {"lat": model_lats_deg, "lon": model_lons_deg},
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


def set_physics_flags(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData = None,
    terrain: TerrainData = None,
) -> tuple[PhysicsTendency, PhysicsData]:
    from jcm.physics.speedy.physical_constants import nstrad

    model_step = physics_data.date.model_step
    compute_shortwave = (jnp.mod(model_step, nstrad) == 0)
    shortwave_data = physics_data.shortwave_rad.copy(compute_shortwave=compute_shortwave)
    physics_data = physics_data.copy(shortwave_rad=shortwave_data)

    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return physics_tendencies, physics_data


class ForcingHumidCloudsRadPhysics(Physics):
    """
    Partial SPEEDY physics:
    forcing + humidity diagnostics + clouds + radiation only.
    """

    parameters: Parameters
    coords: CoordinateSystem
    terms: abc.Sequence[Callable]
    UNITS_TABLE_CSV_PATH = Path(
        "/home/gmathieu/code_uqam/src/jax-gcm/jcm/physics/speedy/units_table.csv"
    )

    def __init__(
        self,
        parameters: Parameters = Parameters.default(),
        checkpoint_terms: bool = True,
    ) -> None:
        self.parameters = parameters

        from jcm.physics.speedy.humidity import spec_hum_to_rel_hum
        from jcm.physics.speedy.shortwave_radiation import (
            get_shortwave_rad_fluxes,
            get_clouds,
        )
        from jcm.physics.speedy.longwave_radiation import (
            get_downward_longwave_rad_fluxes,
            get_upward_longwave_rad_fluxes,
        )
        from jcm.physics.speedy.forcing import set_forcing
        from jcm.physics.speedy.surface_flux import get_surface_fluxes
        from jcm.physics.speedy.vertical_diffusion import get_vertical_diffusion_tend

        physics_terms = [
            set_physics_flags,
            set_forcing,
            spec_hum_to_rel_hum,
            get_clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes,
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend,
        ]

        static_argnums = {
            set_forcing: (2,),
        }

        self.terms = (
            physics_terms
            if not checkpoint_terms
            else [
                jax.checkpoint(
                    term,
                    static_argnums=static_argnums.get(term, ()) + (4,),
                )
                for term in physics_terms
            ]
        )

    def cache_coords(self, coords: CoordinateSystem):
        """Store model coordinate system for SpeedyCoords calculation."""
        self.model_coords = coords
        self.cached_coords = SpeedyCoords.from_coordinate_system(coords)
        return

    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        terrain: TerrainData,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """Compute physical tendencies by looping through selected terms."""
        data = PhysicsData.zeros(
            self.model_coords.horizontal.nodal_shape,
            self.model_coords.nodal_shape[0],
            date=date,
            speedy_coords=self.cached_coords,
        )

        physics_tendency = PhysicsTendency.zeros(shape=state.u_wind.shape)

        model_day_of_year = date.model_day()
        forcing_2d = tree_index_3d(forcing, model_day_of_year)

        for term in self.terms:
            tend, data = term(state, data, self.parameters, forcing_2d, terrain)
            physics_tendency += tend

        return physics_tendency, data

    def get_empty_data(self, coords: CoordinateSystem) -> PhysicsData:
        from jax.tree_util import tree_map

        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        empty_data = PhysicsData.zeros(
            coords.horizontal.nodal_shape,
            coords.nodal_shape[0],
            speedy_coords=speedy_coords,
        )
        empty_data = tree_map(
            lambda x: jnp.zeros_like(x) if hasattr(x, "dtype") else x,
            empty_data,
        )
        empty_data = empty_data.copy(speedy_coords=speedy_coords)
        return empty_data

    def data_struct_to_dict(self, data, nodal_shape):
        """
        Reuse SpeedyPhysics formatting for output diagnostics.
        """
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics

        helper = SpeedyPhysics(parameters=self.parameters, checkpoint_terms=False)
        helper.cache_coords(self.model_coords)
        return helper.data_struct_to_dict(data, nodal_shape=nodal_shape)


def main():
    parser = argparse.ArgumentParser(
        description="Chunked JCM forcing+humid+clouds+rad simulation"
    )
    parser.add_argument(
        "--total_days",
        type=int,
        default=2 * 365,
        help="Total simulation length in days (default: 730)",
    )
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=90,
        help="Days per chunk (default: 90)",
    )
    parser.add_argument(
        "--save_interval",
        type=float,
        default=1.0,
        help="Save interval in days (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        default="jcm_forcing_humid_clouds_rad_sfc_vdiff_run.nc",
        help="Output netCDF path",
    )
    parser.add_argument(
        "--init_time",
        default="2000-01-01T00:00:00",
        help="ERA5 initialisation time (default: 2000-01-01T00:00:00)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Experiment output directory. Final .nc will be saved in outdir/run_nc/",
    )
    args = parser.parse_args()

    total_days = args.total_days
    chunk_days = args.chunk_days
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

    # Setup
    coords = get_speedy_coords()
    data_dir = resources.files("jcm.data.bc.t30.clim")

    realistic_terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)
    realistic_forcing = ForcingData.from_file(data_dir / "forcing.nc", coords=coords)

    params = Parameters.default()
    physics = ForcingHumidCloudsRadPhysics(parameters=params)

    model = Model(
        coords=coords,
        terrain=realistic_terrain,
        physics=physics,
    )

    # ERA5 init
    init_time = pd.Timestamp(args.init_time)
    print(f"Opening ERA5 for initialisation at {init_time} …")

    era5_ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
        consolidated=True,
        storage_options={"token": "anon"},
    ).rename(
        {
            "latitude": "lat",
            "longitude": "lon",
            "u_component_of_wind": "u_wind",
            "v_component_of_wind": "v_wind",
        }
    )

    init_state = build_initial_state(era5_ds, coords, init_time)
    era5_ds.close()
    print("  Initial state built.\n")

    # Chunked integration
    n_chunks = int(np.ceil(total_days / chunk_days))
    chunk_paths = []
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

        if "temperature" in ds_chunk:
            print(
                f"   Saved → {chunk_file}  "
                f"(T range: {float(ds_chunk['temperature'].min()):.1f}–"
                f"{float(ds_chunk['temperature'].max()):.1f} K, "
                f"NaN: {int(ds_chunk['temperature'].isnull().sum())})\n"
            )
        else:
            print(f"   Saved → {chunk_file}\n")

        del predictions, ds_chunk
        remaining -= this_chunk

    # Concatenate
    print(f"Concatenating {len(chunk_paths)} chunks → {out_path}")
    datasets = [xr.open_dataset(p) for p in chunk_paths]
    ds_full = xr.concat(datasets, dim="time")
    ds_full.to_netcdf(out_path)

    for d in datasets:
        d.close()

    print(f"Done. Output: {out_path}")
    print(f"  Time: {ds_full.time.values[0]} → {ds_full.time.values[-1]}")
    print(f"  Shape: {dict(ds_full.sizes)}")
    if "temperature" in ds_full:
        print(
            f"  T range: {float(ds_full['temperature'].min()):.1f}–"
            f"{float(ds_full['temperature'].max()):.1f} K"
        )


if __name__ == "__main__":
    main()
