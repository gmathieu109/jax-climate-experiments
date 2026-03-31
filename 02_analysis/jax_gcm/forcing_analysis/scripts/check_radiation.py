import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 👉 METS ICI ton fichier JCM output (full physics)
file = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/comparisons/full_vs_dry_11y_postspinup_10y/run_nc/jcm_11year_full.nc"

ds = xr.open_dataset(file)

# voir les variables
print(list(ds.data_vars))

# choisir variable radiation (ajuste après print)
da = ds["shortwave_rad.dfabs"]  # ou autre

# moyenne globale
lat = da["lat"]
weights = np.cos(np.deg2rad(lat))

global_mean = da.weighted(weights).mean(dim=("lat","lon"))

# plot
global_mean.plot()
plt.title("Global mean absorbed shortwave")
plt.show()
