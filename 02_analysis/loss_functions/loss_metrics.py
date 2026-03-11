import numpy as np
import xarray as xr


def error(model: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    return model - ref


def remove_initial_bias(err: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    return err - err.isel({time_dim: 0})


def lat_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return xr.DataArray(w, coords={lat.name: lat}, dims=(lat.name,))


def global_bias(
    model: xr.DataArray,
    ref: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    weighted: bool = True,
) -> xr.DataArray:
    err = model - ref

    if not weighted:
        return err.mean(dim=(lat_name, lon_name), skipna=True)

    w = lat_weights(model[lat_name])
    w2 = w.broadcast_like(err)
    num = (err * w2).sum(dim=(lat_name, lon_name), skipna=True)
    den = w2.where(np.isfinite(err)).sum(dim=(lat_name, lon_name), skipna=True)
    return num / den


def global_rmse(
    model: xr.DataArray,
    ref: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon",
    weighted: bool = True,
) -> xr.DataArray:
    err2 = (model - ref) ** 2

    if not weighted:
        return np.sqrt(err2.mean(dim=(lat_name, lon_name), skipna=True))

    w = lat_weights(model[lat_name])
    w2 = w.broadcast_like(err2)
    num = (err2 * w2).sum(dim=(lat_name, lon_name), skipna=True)
    den = w2.where(np.isfinite(err2)).sum(dim=(lat_name, lon_name), skipna=True)
    return np.sqrt(num / den)


def regional_bias(model_box: xr.DataArray, ref_box: xr.DataArray) -> xr.DataArray:
    return (model_box - ref_box).mean(skipna=True)


def regional_rmse(model_box: xr.DataArray, ref_box: xr.DataArray) -> xr.DataArray:
    return np.sqrt(((model_box - ref_box) ** 2).mean(skipna=True))


def point_bias(model_point: xr.DataArray, ref_point: xr.DataArray) -> xr.DataArray:
    return model_point - ref_point


def point_rmse(model_point: xr.DataArray, ref_point: xr.DataArray) -> xr.DataArray:
    return np.sqrt((model_point - ref_point) ** 2)