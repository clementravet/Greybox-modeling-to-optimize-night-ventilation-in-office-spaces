import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product as iter_product
import pvlib


## Step 1 — Open-loop simulation function

def simulate_night_vent_scenario(
    fitted_params,
    X_base,
    N_profile,
    night_start_hour,
    night_stop_hour,
    qv_night,
    rec_duration,
    Ti_init, Tm_init, c_init
):
    """
    Open-loop V9 simulation with modified night ventilation schedule.
    Physics identical to TiTmCn2R2C_summer_V9 — no KF, N taken as input.
    """
    # ── Parameters (all from V9) ──────────────────────────────────────────────
    Ci            = fitted_params['Ci'].value
    Cm            = fitted_params['Cm'].value
    Rim           = fitted_params['Rim'].value
    Rout          = fitted_params['Rout'].value
    q_pers        = fitted_params['q_pers'].value
    q_equip_var   = fitted_params['q_equip_var'].value
    q_equip_const = fitted_params['q_equip_const'].value
    V             = fitted_params['V'].value
    S             = fitted_params['S'].value
    A             = fitted_params['A'].value
    G             = fitted_params['G'].value       # CO2 emission rate
    c_out         = fitted_params['c_out'].value
    rho_air       = fitted_params['rho_air'].value
    cp_air        = fitted_params['cp_air'].value
    gamma_g       = fitted_params['gamma_g'].value
    n             = fitted_params['n'].value
    K             = fitted_params['K'].value
    L             = fitted_params['L'].value
    f_sol         = fitted_params['f_sol'].value   # ← solar split

    q_int_N     = q_pers + q_equip_var
    q_int_const = q_equip_const * S

    # ── Neighbour resistances (auto-detect, same as V9) ───────────────────────
    neigh_keys = sorted(
        [k for k in fitted_params if k.startswith('Rneigh_')],
        key=lambda x: int(x.split('_')[1])
    )
    M      = len(neigh_keys)
    Rneigh = np.array([fitted_params[k].value for k in neigh_keys])
    T_neigh_arr = (
        np.column_stack([X_base[f'T_neigh_{j+1}'] for j in range(M)])
        if M > 0 else None
    )

    # ── Inputs → numpy ────────────────────────────────────────────────────────
    Ta      = np.asarray(X_base['Ta'],      dtype=float)
    Tsup    = np.asarray(X_base['Tsup'],    dtype=float)
    qv      = np.asarray(X_base['qv'],      dtype=float)
    Ik      = np.asarray(X_base['Ik'],      dtype=float)
    theta_z = np.asarray(X_base['theta_z'], dtype=float)
    gamma_s = np.asarray(X_base['gamma_s'], dtype=float)
    num_rec = len(Ta)
    index   = X_base.index

    # ── B-spline solar aperture (identical to V9) ─────────────────────────────
    phi_names = ['phi_a','phi_b','phi_c','phi_d','phi_e',
                 'phi_f','phi_g','phi_h','phi_i','phi_j']
    phi      = np.array([fitted_params[name].value for name in phi_names])
    bsplines = np.column_stack([X_base[f'bs_{i}'] for i in range(len(phi))])

    aoi_deg_all = pvlib.irradiance.aoi(
        surface_tilt=90,
        surface_azimuth=gamma_g,
        solar_zenith=90 - theta_z,
        solar_azimuth=gamma_s
    )
    iam_all = pvlib.iam.physical(aoi_deg_all, n=n, K=K, L=L)
    g_t     = bsplines @ phi   # time-varying solar aperture

    # ── Night ventilation mask ────────────────────────────────────────────────
    hours = np.array([t.hour + t.minute / 60.0 for t in index])
    if night_start_hour > night_stop_hour:        # crosses midnight
        night_mask = (hours >= night_start_hour) | (hours < night_stop_hour)
    else:
        night_mask = (hours >= night_start_hour) & (hours < night_stop_hour)

    # ── Allocate ──────────────────────────────────────────────────────────────
    Ti = np.zeros(num_rec);  Ti[0] = Ti_init
    Tm = np.zeros(num_rec);  Tm[0] = Tm_init
    c  = np.zeros(num_rec);  c[0]  = c_init
    cp_rho_3600 = rho_air * cp_air / 3600.0
    dt          = rec_duration

    # ── Open-loop Euler — V9 physics ─────────────────────────────────────────
    for k in range(1, num_rec):
        N_k = float(N_profile[k-1])

        # Night ventilation override
        if night_mask[k-1] and Ta[k-1] < Ti[k-1]:
            qv_k   = qv_night
            Tsup_k = Ta[k-1]
        else:
            qv_k   = qv[k-1]
            Tsup_k = Tsup[k-1]

        Q_vent    = cp_rho_3600 * qv_k * (Tsup_k - Ti[k-1])
        Q_solar_k = g_t[k-1] * iam_all[k-1] * A * Ik[k-1]   # ← V9 B-spline+IAM
        Q_int     = q_int_N * N_k + q_int_const

        Q_neigh = 0.0
        if M > 0:
            Q_neigh = np.sum((T_neigh_arr[k-1, :] - Ti[k-1]) / Rneigh)

        # Ti — receives f_sol fraction of solar  ← V9 solar split
        dTi = (
            (Tm[k-1] - Ti[k-1]) / (Rim  * Ci)
            + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
            + (Q_vent + Q_int + f_sol * Q_solar_k + Q_neigh) / Ci
        ) * dt
        Ti[k] = Ti[k-1] + dTi

        # Tm — receives (1 - f_sol) fraction of solar  ← V9 solar split
        dTm = (
            (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            + (1.0 - f_sol) * Q_solar_k / Cm
        ) * dt
        Tm[k] = Tm[k-1] + dTm

        # CO2 — full mass balance including occupancy emission  ← fixed
        dc   = (1e6 * (G / V) * N_k - qv_k / V * (c[k-1] - c_out)) * dt
        c[k] = c[k-1] + dc

    return pd.DataFrame({'Ti': Ti, 'Tm': Tm, 'c': c}, index=index)



## Step 2 — Comfort metrics function

def comfort_metrics(Ti_series, index, occ_start=8, occ_end=18,
                    T_comfort_min=22.0, T_comfort_max=26.0):
    """
    Compute comfort metrics during occupied hours only.

    Returns dict with:
      - dh_above   : degree-hours above T_comfort_max
      - dh_below   : degree-hours below T_comfort_min
      - pct_ok     : % of occupied steps within comfort band
      - Ti_peak    : maximum Ti during occupied hours
      - Ti_mean    : mean Ti during occupied hours
    """
    hours       = np.array([t.hour + t.minute / 60.0 for t in index])
    occ_mask    = (hours >= occ_start) & (hours < occ_end)
    Ti_occ      = Ti_series[occ_mask]
    dt_h        = (index[1] - index[0]).seconds / 3600.0   # step in hours

    dh_above = float(np.sum(np.maximum(0, Ti_occ - T_comfort_max)) * dt_h)
    dh_below = float(np.sum(np.maximum(0, T_comfort_min - Ti_occ)) * dt_h)
    pct_ok   = float(np.mean((Ti_occ >= T_comfort_min) & (Ti_occ <= T_comfort_max)) * 100)
    Ti_peak  = float(Ti_occ.max())
    Ti_mean  = float(Ti_occ.mean())

    return dict(dh_above=dh_above, dh_below=dh_below,
                pct_ok=pct_ok, Ti_peak=Ti_peak, Ti_mean=Ti_mean)
