#!/usr/bin/env python3
"""
Optimize a global multiplicative factor on snowc_am for a short full-physics window.

V1:
- optimize only snowc_am
- optional scaffold for alb0 later
- target: ERA5 T850
- loss: weighted RMSE over 5 days
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax

from full_forcing_opt_core import (
    make_full_setup,
    make_modified_forcing,
    run_forward_predictions,
    extract_t850_from_predictions,
    load_era5_target_t850,
    cosine_lat_weights,
    weighted_rmse,
)


def main():
    parser = argparse.ArgumentParser(description="Optimize global snowc_am scale for full JCM")
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
    parser.add_argument("--use_albedo", action="store_true",
                        help="Also optimize alb0 global scale")
    parser.add_argument("--debug_only", action="store_true",
                        help="Run debug checks and stop before optimization")
    args = parser.parse_args()

    print("Building full setup...")
    setup = make_full_setup(init_time=args.init_time)

    print("Preparing ERA5 target T850...")
    target_t850, lat_deg, lon_deg = load_era5_target_t850(
        setup.coords,
        init_time=args.init_time,
        total_days=args.total_days,
    )
    weights = cosine_lat_weights(lat_deg, nlon=len(lon_deg))

    # Parameters in unconstrained space.
    # u=0 gives alpha near 1.
    params = {"u_snow": jnp.array(0.0)}
    if args.use_albedo:
        params["u_alb"] = jnp.array(0.0)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    def loss_fn(param_dict):
        forcing_mod, info = make_modified_forcing(
            setup.forcing,
            u_snow=param_dict["u_snow"],
            use_albedo=args.use_albedo,
            u_alb=param_dict.get("u_alb", None),
        )

        preds = run_forward_predictions(
            setup=setup,
            forcing=forcing_mod,
            total_days=args.total_days,
            save_interval=args.save_interval,
        )

        t850_pred = extract_t850_from_predictions(preds)

        # Safety: clip to common time length if shapes differ by one sample
        nt = min(t850_pred.shape[0], target_t850.shape[0])
        t850_pred_local = t850_pred[:nt]
        t850_ref_local = target_t850[:nt]

        rmse = weighted_rmse(t850_pred_local, t850_ref_local, weights)

        # Small regularization to keep factors near 1
        reg = 1e-3 * (info["alpha_snow"] - 1.0) ** 2
        if args.use_albedo:
            reg = reg + 1e-3 * (info["alpha_alb"] - 1.0) ** 2

        loss = rmse + reg
        return loss, (rmse, info)

    def step(param_dict, opt_state):
        (loss, (rmse, info)), grads = jax.value_and_grad(loss_fn, has_aux=True)(param_dict)
        updates, opt_state = optimizer.update(grads, opt_state, param_dict)
        param_dict = optax.apply_updates(param_dict, updates)
        return param_dict, opt_state, loss, rmse, info, grads

    # ------------------------------------------------------------------
    # Debug check
    # ------------------------------------------------------------------
    print("\nInitial debug check...")

    forcing_mod, info0 = make_modified_forcing(
        setup.forcing,
        u_snow=params["u_snow"],
        use_albedo=args.use_albedo,
        u_alb=params.get("u_alb", None),
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

    print(f"alpha_snow={float(info0['alpha_snow']):.6f}")
    if args.use_albedo:
        print(f"alpha_alb={float(info0['alpha_alb']):.6f}")

    if args.debug_only:
        raise SystemExit("Stopping after debug check because --debug_only was set.")

    # Safety stop if debug is bad
    if bool(jnp.isnan(rmse0)):
        raise SystemExit("Stopping: initial RMSE is NaN. Fix debug issues before optimization.")

    # ------------------------------------------------------------------
    # Initial loss check
    # ------------------------------------------------------------------
    print("\nInitial sensitivity check...")
    loss0, (rmse0_lossfn, info0_lossfn) = loss_fn(params)
    print(
        f"  initial loss={float(loss0):.6f} | "
        f"rmse={float(rmse0_lossfn):.6f} | "
        f"alpha_snow={float(info0_lossfn['alpha_snow']):.6f}"
    )
    if args.use_albedo:
        print(f"  initial alpha_alb={float(info0_lossfn['alpha_alb']):.6f}")

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------
    print("\nOptimization loop...")
    for i in range(args.n_steps):
        params, opt_state, loss, rmse, info, grads = step(params, opt_state)

        msg = (
            f"iter {i:03d} | "
            f"loss={float(loss):.6f} | "
            f"rmse={float(rmse):.6f} | "
            f"alpha_snow={float(info['alpha_snow']):.6f} | "
            f"grad_u_snow={float(grads['u_snow']):.6e}"
        )
        if args.use_albedo:
            msg += (
                f" | alpha_alb={float(info['alpha_alb']):.6f}"
                f" | grad_u_alb={float(grads['u_alb']):.6e}"
            )
        print(msg)

    print("\nDone.")
    print("Final unconstrained params:")
    for k, v in params.items():
        print(f"  {k} = {float(v):.6f}")

    _, info_final = make_modified_forcing(
        setup.forcing,
        u_snow=params["u_snow"],
        use_albedo=args.use_albedo,
        u_alb=params.get("u_alb", None),
    )

    print("Final bounded factors:")
    print(f"  alpha_snow = {float(info_final['alpha_snow']):.6f}")
    if args.use_albedo:
        print(f"  alpha_alb  = {float(info_final['alpha_alb']):.6f}")


if __name__ == "__main__":
    main()
