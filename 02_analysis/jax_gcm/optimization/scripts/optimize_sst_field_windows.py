#!/usr/bin/env python3
"""
Optimize a pixel-wise multiplicative field on albedo for a short full-physics window.
"""

from __future__ import annotations

import argparse
import os
import jax
import jax.numpy as jnp
import optax
import numpy as np


from full_forcing_opt_core import (
    make_full_setup,
    make_modified_forcing,
    run_forward_predictions,
    extract_t850_from_predictions,
    load_era5_target_t850,
    cosine_lat_weights,
    weighted_rmse,
)

INIT_TIMES = [
    "2000-01-01T00:00:00",  # hiver
    "2000-07-01T00:00:00",  # été
]

def main():
    parser = argparse.ArgumentParser(description="Optimize pixel-wise snowc_am scale for full JCM")
    parser.add_argument("--init_time", default="2000-01-01T00:00:00",
                        help="ERA5 init time (default: 2000-01-01T00:00:00)")
    parser.add_argument("--total_days", type=int, default=5,
                        help="Optimization window in days (default: 5)")
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save interval in days (default: 1.0)")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Number of optimizer steps (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate (default: 1e-2)")
    parser.add_argument("--lambda_l2", type=float, default=1e-3,
                        help="L2 regularization on alpha_snow - 1")
    parser.add_argument("--lambda_smooth", type=float, default=1e-3,
                        help="Spatial smoothness regularization")
    parser.add_argument("--debug_only", action="store_true",
                        help="Run debug checks and stop before optimization")
    parser.add_argument(
        "--regional_loss",
        action="store_true",
        help="Restrict RMSE loss to NH extratropics (lat >= 30 deg)"
)
    args = parser.parse_args()

    base_outdir = "/home/gmathieu/extras/outputs_uqam/jcm_experiments/optimisation"

    lr_tag = str(args.lr).replace(".", "p").replace("-", "m")
    def format_lambda(x):
        return str(x).replace(".", "p").replace("-", "m")

    if args.lambda_l2 == 0 and args.lambda_smooth == 0:
        reg_tag = "noreg"
    else:
        l2_tag = format_lambda(args.lambda_l2)
        sm_tag = format_lambda(args.lambda_smooth)
        reg_tag = f"l2_{l2_tag}_sm_{sm_tag}"
    scope_tag = "nh30" if args.regional_loss else "global"

    outdir = os.path.join(
        base_outdir,
        f"delta_sst_{args.total_days}d_steps{args.n_steps}_lr{lr_tag}_{reg_tag}_{scope_tag}"
    )

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    print("Building full setup...")
    setup = make_full_setup(init_time=args.init_time)


    print("Preparing ERA5 target T850...")
    target_t850, lat_deg, lon_deg = load_era5_target_t850(
        setup.coords,
        init_time=args.init_time,
        total_days=args.total_days,
    )
    weights = cosine_lat_weights(lat_deg, nlon=len(lon_deg))

    if args.regional_loss:
        lat_mask = (lat_deg >= 30.0).astype(np.float32)   # shape (lat,)
        region_mask_2d = np.tile(lat_mask[None, :], (len(lon_deg), 1))  # (lon, lat)
        weights = weights * jnp.asarray(region_mask_2d)

        # renormalize only over selected region to keep scale reasonable
        wmean = jnp.mean(weights)
        weights = jnp.where(wmean > 0, weights / wmean, weights)

        print("Using REGIONAL loss: NH extratropics only (lat >= 30 deg)")
        print("Selected latitude count:", int(lat_mask.sum()), "/", len(lat_deg))
    else:
        print("Using GLOBAL loss")

    print("weights min/max/mean:",
      float(jnp.min(weights)),
      float(jnp.max(weights)),
      float(jnp.mean(weights)))

    # Pixel-wise unconstrained field on the horizontal grid
    params = {
        "u_sst": jnp.zeros_like(setup.forcing.alb0)
    }

    optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(args.lr),
    )
    opt_state = optimizer.init(params)

    def spatial_smoothness(field_2d: jnp.ndarray) -> jnp.ndarray:
        reg_lon = jnp.mean((field_2d[1:, :] - field_2d[:-1, :]) ** 2)
        reg_lat = jnp.mean((field_2d[:, 1:] - field_2d[:, :-1]) ** 2)
        return reg_lon + reg_lat

def loss_fn(params):
    u_sst = params["u_sst"]
    delta_sst = 2.0 * jnp.tanh(u_sst)

    total_rmse = 0.0

    # ------------------------
    # Loop over init times
    # ------------------------
    for init_time in INIT_TIMES:
        setup = make_full_setup(init_time=init_time)

        target_t850, _, _ = load_era5_target_t850(
            setup.coords,
            init_time=init_time,
            total_days=args.total_days,
        )

        dummy_u_snow = jnp.zeros_like(setup.forcing.alb0)

        forcing_mod, _ = make_modified_forcing(
            setup.forcing,
            u_snow=dummy_u_snow,
            use_sst=True,
            u_sst=delta_sst,
        )

        preds = run_forward_predictions(
            setup=setup,
            forcing=forcing_mod,
            total_days=args.total_days,
            save_interval=1.0,
        )

        t850_pred = extract_t850_from_predictions(preds)

        diff = t850_pred - target_t850
        rmse = jnp.sqrt(jnp.mean(diff**2))

        total_rmse += rmse

    # ------------------------
    # Average RMSE (IMPORTANT: outside loop)
    # ------------------------
    rmse = total_rmse / len(INIT_TIMES)

    # ------------------------
    # Regularization (ONLY ONCE)
    # ------------------------
    reg_l2 = jnp.mean(delta_sst**2)

    dx = delta_sst[1:, :] - delta_sst[:-1, :]
    dy = delta_sst[:, 1:] - delta_sst[:, :-1]
    reg_smooth = jnp.mean(dx**2) + jnp.mean(dy**2)

    # ------------------------
    # Final loss
    # ------------------------
    loss = rmse + args.lambda_l2 * reg_l2 + args.lambda_smooth * reg_smooth

    info = {"delta_sst": delta_sst}

    return loss, (rmse, reg_l2, reg_smooth, info)

    def step(param_dict, opt_state):
        (loss, (rmse, reg_l2, reg_smooth, info)), grads = jax.value_and_grad(loss_fn, has_aux=True)(param_dict)
        updates, opt_state = optimizer.update(grads, opt_state, param_dict)
        param_dict = optax.apply_updates(param_dict, updates)
        return param_dict, opt_state, loss, rmse, reg_l2, reg_smooth, info, grads

    # ------------------------
    # Debug
    # ------------------------
    print("\nInitial debug check...")

    dummy_u_snow = jnp.zeros_like(setup.forcing.alb0)

    forcing_mod, info0 = make_modified_forcing(
        setup.forcing,
        u_snow=dummy_u_snow,
        use_albedo=False,
        use_sst=True,
        u_sst=2.0 * jnp.tanh(params["u_sst"]),
    )

    preds = run_forward_predictions(
        setup=setup,
        forcing=forcing_mod,
        total_days=args.total_days,
        save_interval=args.save_interval,
    )

    t850_pred = extract_t850_from_predictions(preds)

    print("t850_pred shape:", t850_pred.shape)
    print("target_t850 shape:", target_t850.shape)
    print("t850_pred has nan:", bool(jnp.isnan(t850_pred).any()))
    print("target_t850 has nan:", bool(jnp.isnan(target_t850).any()))
    print("t850_pred min/max:", float(jnp.nanmin(t850_pred)), float(jnp.nanmax(t850_pred)))
    print("target_t850 min/max:", float(jnp.nanmin(target_t850)), float(jnp.nanmax(target_t850)))

    nt = min(t850_pred.shape[0], target_t850.shape[0])
    t850_pred_dbg = t850_pred[:nt]
    t850_ref_dbg = target_t850[:nt]

    diff = t850_pred_dbg - t850_ref_dbg
    print("diff has nan:", bool(jnp.isnan(diff).any()))
    print("diff min/max:", float(jnp.nanmin(diff)), float(jnp.nanmax(diff)))

    rmse0 = weighted_rmse(t850_pred_dbg, t850_ref_dbg, weights)
    print("rmse0:", float(rmse0))
    delta = info0["delta_sst"]

    print("delta_sst mean/min/max:",
        float(jnp.mean(delta)),
        float(jnp.min(delta)),
        float(jnp.max(delta)))

    if args.debug_only:
        raise SystemExit("Stopping after debug check because --debug_only was set.")

    if bool(jnp.isnan(rmse0)):
        raise SystemExit("Stopping: initial RMSE is NaN. Fix debug issues before optimization.")

    # ------------------------
    # Optimization
    # ------------------------
    metrics = {
        "iter": [],
        "rmse": [],
        "loss": [],
        "delta_mean": [],
        "delta_min": [],
        "delta_max": [],
    }

    print("\nOptimization loop...")
    for i in range(args.n_steps):
        params, opt_state, loss, rmse, reg_l2, reg_smooth, info, grads = step(params, opt_state)

        print("grad has nan:", bool(jnp.isnan(grads["u_sst"]).any()))
        print("grad has inf:", bool(jnp.isinf(grads["u_sst"]).any()))
        print("grad min/max:", float(jnp.nanmin(grads["u_sst"])), float(jnp.nanmax(grads["u_sst"])))

        grad_norm = float(jnp.sqrt(jnp.mean(grads["u_sst"] ** 2)))
        delta = info["delta_sst"]

        metrics["iter"].append(i)
        metrics["rmse"].append(float(rmse))
        metrics["loss"].append(float(loss))
        metrics["delta_mean"].append(float(jnp.mean(delta)))
        metrics["delta_min"].append(float(jnp.min(delta)))
        metrics["delta_max"].append(float(jnp.max(delta)))

        print(
            f"iter {i:03d} | "
            f"loss={float(loss):.6f} | "
            f"rmse={float(rmse):.6f} | "
            f"reg_l2={float(reg_l2):.6f} | "
            f"reg_smooth={float(reg_smooth):.6f} | "
            f"delta_mean={float(jnp.mean(delta)):.6f} | "
            f"delta_min={float(jnp.min(delta)):.6f} | "
            f"delta_max={float(jnp.max(delta)):.6f} | "
            f"grad_rms={grad_norm:.6e}"
        )

        if i % 5 == 0:
            ckpt_file = os.path.join(outdir, f"delta_sst_step_{i:03d}.npy")
            np.save(ckpt_file, np.asarray(info["delta_sst"]))
            print("Saved checkpoint:", ckpt_file)
        if not np.isfinite(float(loss)) or not np.isfinite(float(rmse)) or not np.isfinite(grad_norm):
            print(f"Stopping early at iter {i}: non-finite value detected.")
            break
    print("\nDone.")

    dummy_u_snow = jnp.zeros_like(setup.forcing.alb0)

    delta_sst_final = 2.0 * jnp.tanh(params["u_sst"])

    outfile = os.path.join(outdir, "delta_sst_final.npy")
    np.save(outfile, np.asarray(delta_sst_final))

    print("Saved delta_sst_final.npy to:", outfile)

    print("Final delta_sst summary:")
    print(f"  mean = {float(jnp.mean(delta_sst_final)):.6f}")
    print(f"  min  = {float(jnp.min(delta_sst_final)):.6f}")
    print(f"  max  = {float(jnp.max(delta_sst_final)):.6f}")

    metrics_file = os.path.join(outdir, "metrics.npy")
    np.save(metrics_file, metrics)
    print("Saved metrics.npy to:", metrics_file)

if __name__ == "__main__":
    main()