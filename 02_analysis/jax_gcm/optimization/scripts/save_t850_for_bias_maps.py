from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp

from full_forcing_opt_core import (
    make_full_setup,
    make_modified_forcing,
    run_forward_predictions,
    extract_t850_from_predictions,
    load_era5_target_t850,
)

OUTDIR = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/optimisation/alpha_albedo_regional_nh30"
ALPHA_DIR = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/optimisation/alpha_albedo_30d_steps5_lr0p003_noreg_global"
EVAL_DIR = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/optimisation/alpha_albedo_eval_365d_from_30d_lr0p003"
INIT_TIME = "2000-01-01T00:00:00"
TOTAL_DAYS = 365
SAVE_INTERVAL = 1.0


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("Building setup...")
    setup = make_full_setup(init_time=INIT_TIME)

    print("Loading optimized alpha_albedo field...")
    alpha_path = os.path.join(ALPHA_DIR, "alpha_alb_final.npy")
    alpha_final = np.load(alpha_path)  # shape (lon, lat)

    print("alpha_final shape:", alpha_final.shape)
    print("alpha_final min/max/mean:",
          float(alpha_final.min()),
          float(alpha_final.max()),
          float(alpha_final.mean()))

    # Convert alpha_snow back to u_snow so we can reuse make_modified_forcing(...)
    # alpha = 0.5 + sigmoid(u)
    # => sigmoid(u) = alpha - 0.5
    # => u = log(p / (1-p)), with p = alpha - 0.5
    p = np.clip((alpha_final - 0.8) / 0.4, 1e-6, 1 - 1e-6)
    u_albedo_final = np.log(p / (1.0 - p))

    print("Loading ERA5 target T850...")
    target_t850, lat_deg, lon_deg = load_era5_target_t850(
        setup.coords,
        init_time=INIT_TIME,
        total_days=TOTAL_DAYS,
    )

    print("Running initial/original forcing...")
    preds_init = run_forward_predictions(
        setup=setup,
        forcing=setup.forcing,
        total_days=TOTAL_DAYS,
        save_interval=SAVE_INTERVAL,
    )
    t850_init = extract_t850_from_predictions(preds_init)

    print("Building modified forcing with optimized alpha...")
    dummy_u_snow = jnp.zeros_like(setup.forcing.alb0)

    forcing_final, info_final = make_modified_forcing(
        setup.forcing,
        u_snow=dummy_u_snow,
        use_albedo=True,
        u_alb=jnp.asarray(u_albedo_final),
    )
    

    print("Running final/optimized forcing...")
    preds_final = run_forward_predictions(
        setup=setup,
        forcing=forcing_final,
        total_days=TOTAL_DAYS,
        save_interval=SAVE_INTERVAL,
    )
    t850_final = extract_t850_from_predictions(preds_final)

    print("Shapes:")
    print("t850_init  :", t850_init.shape)
    print("t850_final :", t850_final.shape)
    print("target_t850:", target_t850.shape)

    np.save(os.path.join(EVAL_DIR, "t850_init.npy"), np.asarray(t850_init))
    np.save(os.path.join(EVAL_DIR, "t850_final.npy"), np.asarray(t850_final))
    np.save(os.path.join(EVAL_DIR, "t850_target.npy"), np.asarray(target_t850))
    np.save(os.path.join(EVAL_DIR, "lat_deg.npy"), np.asarray(lat_deg))
    np.save(os.path.join(EVAL_DIR, "lon_deg.npy"), np.asarray(lon_deg))

    print("Saved:")
    print(os.path.join(EVAL_DIR, "t850_init.npy"))
    print(os.path.join(EVAL_DIR, "t850_final.npy"))
    print(os.path.join(EVAL_DIR, "t850_target.npy"))
    print(os.path.join(EVAL_DIR, "lat_deg.npy"))
    print(os.path.join(EVAL_DIR, "lon_deg.npy"))


if __name__ == "__main__":
    main()
