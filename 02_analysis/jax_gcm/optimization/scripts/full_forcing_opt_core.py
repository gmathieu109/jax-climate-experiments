#!/usr/bin/env python3
"""
Core utilities for full-physics forcing optimization experiments.

V1:
- full physics only
- optimize global multiplicative factor on snowc_am
- 5-day windows
- target variable: T850
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import PhysicsState
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.params import Parameters


ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)

JCM_SIGMA_850 = 0.835


@dataclass
class FullSetup:
    coords: object
    terrain: TerrainData
    forcing: ForcingData
    init_state: PhysicsState
    init_time: pd.Timestamp


def build_initial_state(era5_ds: xr.Dataset, coords, init_time: pd.Timestamp) -> PhysicsState:
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


def load_era5_dataset() -> xr.Dataset:
    """Open the same low-res ERA5 dataset used in your run/eval scripts."""
    return xr.open_zarr(
        ERA5_ZARR,
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


def make_full_setup(init_time: str = "2000-01-01T00:00:00") -> FullSetup:
    """Load coords, terrain, forcing, and initial state exactly like the full run."""
    coords = get_speedy_coords()
    data_dir = resources.files("jcm.data.bc.t30.clim")

    terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)
    forcing = ForcingData.from_file(data_dir / "forcing.nc", coords=coords)

    init_time_ts = pd.Timestamp(init_time)

    era5_ds = load_era5_dataset()
    init_state = build_initial_state(era5_ds, coords, init_time_ts)
    era5_ds.close()

    return FullSetup(
        coords=coords,
        terrain=terrain,
        forcing=forcing,
        init_state=init_state,
        init_time=init_time_ts,
    )


def build_model(coords, terrain) -> Model:
    """Construct a fresh full-physics model."""
    params = Parameters.default()
    return Model(
        coords=coords,
        terrain=terrain,
        physics=SpeedyPhysics(parameters=params),
    )


def bounded_alpha_snow(u_snow: jnp.ndarray) -> jnp.ndarray:
    """
    Map unconstrained 2D field u_snow -> alpha_snow in [0.5, 1.5].
    alpha=1 at u=0.
    """
    return 0.5 + jax.nn.sigmoid(u_snow)


def bounded_alpha_alb(u_alb: jnp.ndarray) -> jnp.ndarray:
    """
    Optional second parameter later.
    Map unconstrained scalar u_alb -> alpha_alb in [0.8, 1.2].
    alpha=1 at u=0.
    """
    return 0.8 + 0.4 * jax.nn.sigmoid(u_alb)


def make_modified_forcing(
    forcing: ForcingData,
    u_snow: jnp.ndarray,
    use_albedo: bool = False,
    u_alb: jnp.ndarray | None = None,
    use_sst: bool = False,
    u_sst: jnp.ndarray | None = None,
) -> tuple[ForcingData, dict]:
    """
    Return forcing with pixel-wise scaled snowc_am, and optionally alb0.
    """
    alpha_snow = bounded_alpha_snow(u_snow)
    #Ici, [..., None] = [:, :, None] pour faire du broadcasting sur les dimensions lon, lat
    snowc_new = jnp.clip(alpha_snow[..., None] * forcing.snowc_am, 0.0, 20000.0)

    forcing_new = forcing.copy(snowc_am=snowc_new)

    info = {"alpha_snow": alpha_snow}

    if use_albedo:
        if u_alb is None:
            raise ValueError("use_albedo=True but u_alb is None")
        alpha_alb = bounded_alpha_alb(u_alb)
        alb0_new = jnp.clip(alpha_alb * forcing.alb0, 0.0, 1.0)
        forcing_new = forcing_new.copy(alb0=alb0_new)
        info["alpha_alb"] = alpha_alb

    if use_sst:
        if u_sst is None:
            raise ValueError("use_sst=True but u_sst is None")

        # shape: (lon, lat, time)
        sst = forcing.sea_surface_temperature

        # broadcast (lon, lat) → (lon, lat, time)
        sst_new = sst + u_sst[:, :, None]

        forcing_new = forcing_new.copy(
            sea_surface_temperature=sst_new
        )

        info["delta_sst"] = u_sst

    return forcing_new, info


def run_forward_predictions(
    setup: FullSetup,
    forcing: ForcingData,
    total_days: int = 5,
    save_interval: float = 1.0,
):
    """
    Run a fresh full-physics model forward with the supplied forcing.

    Returns the raw Predictions object from model.run().
    """
    model = build_model(setup.coords, setup.terrain)
    preds = model.run(
        initial_state=setup.init_state,
        forcing=forcing,
        total_time=total_days,
        save_interval=save_interval,
    )
    return preds


def extract_t850_from_predictions(preds) -> jnp.ndarray:
    """
    Extract T850-ish directly from the JAX Predictions object.

    Expected shape of preds.dynamics.temperature:
        (time, level, lon, lat)

    Returns:
        T850 with shape (time, lon, lat)
    """
    temp = preds.dynamics.temperature
    # temp shape is expected to be (time, level, lon, lat)
    level_idx = int(np.argmin(np.abs(np.asarray(get_speedy_coords().vertical.centers) - JCM_SIGMA_850)))
    return temp[:, level_idx, :, :]


def load_era5_target_t850(
    coords,
    init_time: str = "2000-01-01T00:00:00",
    total_days: int = 5,
) -> tuple[jnp.ndarray, np.ndarray, np.ndarray]:
    """
    Build a daily ERA5 T850 target on the JCM grid for the optimization window.

    Returns:
        target_t850: jnp.ndarray with shape (time, lon, lat)
        lat_deg: np.ndarray
        lon_deg: np.ndarray
    """
    init_time_ts = pd.Timestamp(init_time)
    end_time_ts = init_time_ts + pd.Timedelta(days=total_days - 1)

    era5_ds = load_era5_dataset()

    model_lats_deg = coords.horizontal.latitudes * 180 / np.pi
    model_lons_deg = coords.horizontal.longitudes * 180 / np.pi

    t850 = era5_ds["temperature"].sel(
        level=850,
        time=slice(init_time_ts, end_time_ts + pd.Timedelta(hours=18)),
    )

    if t850.lat.values[0] > t850.lat.values[-1]:
        t850 = t850.sortby("lat")

    t850_interp = t850.interp(
        lat=model_lats_deg,
        lon=model_lons_deg,
        method="linear",
    )

    # Fill any interpolation gaps
    t850_interp = t850_interp.interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
    t850_interp = t850_interp.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

    # Convert 6-hourly ERA5 to daily mean
    t850_daily = t850_interp.resample(time="1D").mean()

    # Safety fill in case any NaNs remain after daily mean
    t850_daily = t850_daily.interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
    t850_daily = t850_daily.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

    t850_daily = t850_daily.transpose("time", "lon", "lat")

    return (
        jnp.asarray(t850_daily.values),
        np.asarray(model_lats_deg),
        np.asarray(model_lons_deg),
    )


def cosine_lat_weights(lat_deg: np.ndarray, nlon: int) -> jnp.ndarray:
    """
    Build 2D cos(lat) weights with shape (lon, lat) to match JCM arrays.
    """
    w_lat = np.cos(np.deg2rad(lat_deg))
    w_lat = w_lat / w_lat.mean()
    w_2d = np.tile(w_lat[None, :], (nlon, 1))
    return jnp.asarray(w_2d)


def weighted_rmse(pred: jnp.ndarray, target: jnp.ndarray, weights_2d: jnp.ndarray) -> jnp.ndarray:
    """
    pred, target: (time, lon, lat)
    weights_2d:   (lon, lat)
    """
    err2 = (pred - target) ** 2
    weighted = err2 * weights_2d[None, :, :]
    return jnp.sqrt(jnp.mean(weighted))
