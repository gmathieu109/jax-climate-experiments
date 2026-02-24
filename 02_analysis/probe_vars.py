import xarray as xr

FULL="/home/gmathieu/links/scratch/jaxgcm_outputs/pred_ds_jax_gcm_full_physics_2days_12h.nc"
DINO="/home/gmathieu/links/scratch/dinosaur_outputs/ds_out_dino_2days_12h.nc"
ERA="/home/gmathieu/links/scratch/era5_snapshot/era5_light2d_19900501T00.nc"

def show(path, name):
    ds = xr.open_dataset(path)
    print(f"\n===== {name} =====")
    print("path:", path)
    print("coords:", list(ds.coords))
    print("dims:", dict(ds.dims))
    print("\nDATA_VARS (name | dims | units):")
    for v in sorted(ds.data_vars):
        units = ds[v].attrs.get("units","")
        print(f"  {v:45s} {str(ds[v].dims):28s} units={units}")
    # try common lat/lon detection
    for cand in ["lat","latitude","y"]:
        if cand in ds.coords or cand in ds.variables:
            print("lat candidate:", cand, "ndim=", ds[cand].ndim)
            break
    for cand in ["lon","longitude","x"]:
        if cand in ds.coords or cand in ds.variables:
            print("lon candidate:", cand, "ndim=", ds[cand].ndim)
            break
    ds.close()

show(FULL,"FULL")
show(DINO,"DINO")
show(ERA,"ERA5")
