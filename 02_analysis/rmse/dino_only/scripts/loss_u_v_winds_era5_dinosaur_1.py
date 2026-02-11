import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DINO_PATH = "/scratch/gmathieu/dinosaur_outputs/ds_out_small.nc"
OUTDIR = "/scratch/gmathieu/dinosaur_outputs/rmse_uv"
os.makedirs(OUTDIR, exist_ok=True)

T0 = np.datetime64("1990-05-01T00:00:00")

# Dino coords (d'après ton erreur)
DINO_LAT = "latitude"
DINO_LON = "longitude"
DINO_TIME = "time"
DINO_LEV = "sigma"

# Ajuste si tes variables dino ne s'appellent pas exactement comme ça
DINO_U = "u_component_of_wind" if "u_component_of_wind" else "u"
DINO_V = "v_component_of_wind" if "v_component_of_wind" else "v"

ERA5_ML_ZARR = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
ERA_U = "u_component_of_wind"
ERA_V = "v_component_of_wind"


def open_dino(path):
    ds = xr.open_dataset(path, engine="scipy")  # NetCDF3 64-bit offset
    # time float -> datetime64 (si ton time est en secondes)
    if np.issubdtype(ds[DINO_TIME].dtype, np.number):
        ds = ds.assign_coords({
            DINO_TIME: (
                T0 + (ds[DINO_TIME] * 3600).astype("timedelta64[s]")
            ).astype("datetime64[ns]")
        })


    # lon 0-360 + tri
    if ds[DINO_LON].min() < 0:
        ds = ds.assign_coords({DINO_LON: (ds[DINO_LON] % 360)})
    ds = ds.sortby(DINO_LON)
    ds = ds.sortby(DINO_LAT)
    return ds


def open_era5_ml():
    import xarray as xr
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
        storage_options=dict(token="anon"),
        consolidated=True,   # ✅ lit .zmetadata (beaucoup moins lourd)
        chunks=None,         # ✅ PAS de dask -> pas de gros graph
    )[["u_component_of_wind", "v_component_of_wind"]]  # ✅ n'ouvre que ce dont tu as besoin

    # lon 0–360 + tri
    if ds["longitude"].min() < 0:
        ds = ds.assign_coords(longitude=(ds["longitude"] % 360))
    ds = ds.sortby("longitude").sortby("latitude")
    return ds

def nearest_time_index(t_era, t):
    j = np.searchsorted(t_era, t)
    if j == 0:
        return 0
    if j >= len(t_era):
        return len(t_era) - 1
    return j if (t_era[j] - t) < (t - t_era[j-1]) else (j - 1)


def rmse2d(a_np, b_np):
    d = (a_np.astype(np.float32) - b_np.astype(np.float32))
    return float(np.sqrt(np.mean(d * d)))




def main():
    ds_d = open_dino(DINO_PATH)
    ds_e = open_era5_ml()
    t_era = ds_e["time"].values.astype("datetime64[ns]")  # charge juste l'index temps (OK)


    # --- assure-toi des noms u/v dino ---
    dvars = set(ds_d.data_vars)
    if DINO_U not in dvars or DINO_V not in dvars:
        # fallback simple: cherche des noms contenant "u" / "v"
        print("Dino data_vars:", list(ds_d.data_vars))
        raise KeyError(f"Vars dino introuvables: {DINO_U}, {DINO_V}")

    # niveau bas dino
    du_all = ds_d[DINO_U]
    dv_all = ds_d[DINO_V]
    if DINO_LEV in du_all.dims:
        du_all = du_all.isel({DINO_LEV: -1})
    if DINO_LEV in dv_all.dims:
        dv_all = dv_all.isel({DINO_LEV: -1})

    times = ds_d[DINO_TIME].values
    rmse_u = np.empty(len(times), dtype=np.float32)
    rmse_v = np.empty(len(times), dtype=np.float32)
    rmse_spd = np.empty(len(times), dtype=np.float32)

    dlat = ds_d[DINO_LAT]
    dlon = ds_d[DINO_LON]

    for i, t in enumerate(times):

        # -------------------------------------------------
        # 1) index temps ERA5 le plus proche
        # -------------------------------------------------
        j = nearest_time_index(t_era, np.datetime64(t))
        t_e = t_era[j]

        # -------------------------------------------------
        # 2) Dino à ce temps (DataArray)
        # -------------------------------------------------
        du_da = du_all.isel({DINO_TIME: i})
        dv_da = dv_all.isel({DINO_TIME: i})

        # -------------------------------------------------
        # 3) ERA5 u/v → niveau bas → interp sur grille dino
        # -------------------------------------------------
        eu_da = (
            ds_e[ERA_U]
            .isel(time=j, hybrid=-1)
            .interp(
                latitude=ds_d[DINO_LAT],
                longitude=ds_d[DINO_LON],
                method="linear",
            )
            .rename({"latitude": DINO_LAT, "longitude": DINO_LON})
        )

        ev_da = (
            ds_e[ERA_V]
            .isel(time=j, hybrid=-1)
            .interp(
                latitude=ds_d[DINO_LAT],
                longitude=ds_d[DINO_LON],
                method="linear",
            )
            .rename({"latitude": DINO_LAT, "longitude": DINO_LON})
        )

        # -------------------------------------------------
        # 4) FORCER MEME ORDRE DES AXES
        # -------------------------------------------------
        du = du_da.transpose(DINO_LAT, DINO_LON).values.astype(np.float32)
        dv = dv_da.transpose(DINO_LAT, DINO_LON).values.astype(np.float32)
        eu = eu_da.transpose(DINO_LAT, DINO_LON).values.astype(np.float32)
        ev = ev_da.transpose(DINO_LAT, DINO_LON).values.astype(np.float32)

        # -------------------------------------------------
        # 5) RMSE u / v
        # -------------------------------------------------
        rmse_u[i] = rmse2d(du, eu)
        rmse_v[i] = rmse2d(dv, ev)

        # -------------------------------------------------
        # 6) RMSE wind speed
        # -------------------------------------------------
        dspd = np.sqrt(du * du + dv * dv)
        espd = np.sqrt(eu * eu + ev * ev)
        rmse_spd[i] = rmse2d(dspd, espd)

        # -------------------------------------------------
        # 7) logs
        # -------------------------------------------------
        if i % 10 == 0:
            print(
                f"{i:4d}/{len(times)}  "
                f"t_model={t}  t_era5={t_e}  "
                f"rmse_u={rmse_u[i]:.3f}"
            )

        if i == 0:
            print("du dims/shape:", du_da.dims, du_da.shape)
            print("eu dims/shape:", eu_da.dims, eu_da.shape)
            print("numpy shapes:", du.shape, eu.shape)


    # sauvegarde netcdf léger
    out = xr.Dataset(
        {
            "rmse_u": (("time",), rmse_u),
            "rmse_v": (("time",), rmse_v),
            "rmse_wind_speed": (("time",), rmse_spd),
        },
        coords={"time": times},
    )
    out.to_netcdf(
    os.path.join(OUTDIR, "rmse_uv_speed.nc"),
    engine="scipy",
    format="NETCDF3_64BIT",
    )


    # png
    def plot_series(y, title, fname):
        plt.figure()
        plt.plot(times, y)
        plt.title(title)
        plt.xlabel("time (h)")
        plt.ylabel("RMSE (m/s)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
        plt.close()

    plot_series(rmse_u, "RMSE(u) vs ERA5 (lowest level)", "rmse_u.png")
    plot_series(rmse_v, "RMSE(v) vs ERA5 (lowest level)", "rmse_v.png")
    plot_series(rmse_spd, "RMSE(wind speed) vs ERA5 (lowest level)", "rmse_wind_speed.png")

    print("Done. Outputs in:", OUTDIR)


if __name__ == "__main__":
    main()

