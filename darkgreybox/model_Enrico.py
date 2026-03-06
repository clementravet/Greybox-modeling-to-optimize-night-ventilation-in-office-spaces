from ast import Param
import numpy as np
import pvlib

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult


class TiTmCn2R2C_summer_V12(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)
      - Full multivariate Kalman Filter on [Ti, Tm, N]
        replacing the explicit Euler ODE with an SDE solver

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta       : Ambient temperature (°C)
    Tsup     : Supply air temperature for ventilation (°C)
    qv       : Ventilation flow rate (m3/h)
    Ik       : Irradiance (W/m²)
    c        : CO2 concentration (ppm) [measured]
    Ti_meas  : Measured indoor temperature (°C)

    Parameters (params)
    -------------------
    Ti0, Tm0, c0, N0  : Initial states
    Ci                : Air + light capacitance (J/K)
    Cm                : Thermal mass capacitance (J/K)
    Rim               : Resistance Ti-Tm (K/W)
    Rout              : Resistance Ti-Ta (K/W)
    rho_air, cp_air   : Air density (kg/m3), specific heat (J/kgK)
    V                 : Room volume (m³)
    S                 : Room surface (m²)
    A                 : Window area (m²)
    G_base            : CO2 emission per person at 1 Met (m³/h/person), fixed=0.016
    Met               : Metabolic rate (Met), fixed=1.2
    c_out             : Outdoor CO2 fraction (ppm)
    alpha_lat         : Latent heat per person (W/person)
    q_equip_var       : Equipment heat gain per person (W/person)
    q_equip_const     : Constant equipment gains (W/m²)
    g                 : Solar transmittance of glazing
    sigma_Ti          : Process noise std dev for Ti [K/step]
    sigma_Tm          : Process noise std dev for Tm [K/step]
    sigma_N           : Process noise std dev for N  [persons/step]
    sigma_Ti_meas     : Ti sensor noise std dev [K]    — fixed from spec
    sigma_c           : CO2 sensor noise std dev [ppm] — fixed from spec
    P0_Ti, P0_Tm, P0_N: Initial state variances
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # ── Allocate ──────────────────────────────────────────────────────
        Ti      = np.zeros(num_rec)
        Tm      = np.zeros(num_rec)
        c       = np.zeros(num_rec)
        N       = np.zeros(num_rec)
        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        z_Ti    = np.zeros(num_rec)
        S_Ti    = np.ones(num_rec)

        # ── Initial conditions ────────────────────────────────────────────
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # ── Physical parameters ───────────────────────────────────────────
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        g             = params['g'].value
        Met           = params['Met'].value
        G_base        = params['G_base'].value
        alpha_lat     = params['alpha_lat'].value

        G      = G_base * Met
        q_sens = Met * (100.0 - alpha_lat)

        # ── KF noise parameters ───────────────────────────────────────────
        sigma_Ti      = params['sigma_Ti'].value
        sigma_Tm      = params['sigma_Tm'].value
        sigma_N       = params['sigma_N'].value
        sigma_Ti_meas = params['sigma_Ti_meas'].value
        sigma_c       = params['sigma_c'].value

        Q_mat = np.diag([sigma_Ti**2, sigma_Tm**2, sigma_N**2])

        # ── KF initial state and covariance ───────────────────────────────
        x_hat = np.array([Ti[0], Tm[0], float(params['N0'])])
        P     = np.diag([params['P0_Ti'].value,
                         params['P0_Tm'].value,
                         params['P0_N'].value])
        I3    = np.eye(3)

        # ── Inputs → numpy (avoids pandas per-element overhead) ──────────
        Ta      = np.asarray(X['Ta'],      dtype=float)
        Tsup    = np.asarray(X['Tsup'],    dtype=float)
        qv      = np.asarray(X['qv'],      dtype=float)
        Ik      = np.asarray(X['Ik'],      dtype=float)
        c_meas  = np.asarray(X['c'],       dtype=float)
        Ti_meas = np.asarray(X['Ti_meas'], dtype=float)

        dt = self.rec_duration

        # ── Precompute scalars used every step ────────────────────────────
        cp_rho_3600 = rho_air * cp_air / 3600.0     # rho*cp/3600
        gA          = g * A                         # solar gain factor
        GV_1e6      = (G / V) * 1e6                 # CO2 emission factor
        q_int_N     = q_sens + q_equip_var          # heat gain per person
        q_int_const = q_equip_const * S             # constant heat gain

        # ── F is constant (depends only on params, not on k) ─────────────
        # Precompute ONCE — avoids rebuilding a 3x3 array every iteration
        F = np.array([
            [1.0 - dt/(Rim*Ci) - dt/(Rout*Ci),
                    dt/(Rim*Ci),
                    q_int_N * dt / Ci],
            [dt/(Rim*Cm),
                    1.0 - dt/(Rim*Cm),
                    0.0],
            [0.0, 
                    0.0, 
                    1.0]
        ])
        FT = F.T                                    # precompute transpose too

        # ── Update 1 constants: H_Ti = [1, 0, 0] ─────────────────────────
        # H_Ti @ P = P[0, :]  →  S = P[0,0] + R,  K = P[:,0] / S
        R_Ti = sigma_Ti_meas ** 2

        # ── Update 2 constants: H_c = [0, 0, hc2] ────────────────────────
        # H_c @ P = hc2 * P[2, :]  →  S = hc2²*P[2,2] + R,  K = P[:,2]*hc2/S
        hc2 = GV_1e6 * dt
        R_c = sigma_c ** 2

        # ── Main loop ─────────────────────────────────────────────────────
        for k in range(1, num_rec):

            # Deterministic inputs
            Q_vent_k  = cp_rho_3600 * qv[k-1] * (Tsup[k-1] - x_hat[0])
            Q_solar_k = gA * Ik[k-1]
            Q_vent[k]  = Q_vent_k
            Q_solar[k] = Q_solar_k

            # b[0] only (b[1]=b[2]=0), added directly to x_pred[0]
            b0 = (Q_vent_k + Q_solar_k + q_int_const) * dt / Ci

            # ── PREDICT ───────────────────────────────────────────────────
            x_pred    = F @ x_hat
            x_pred[0] += b0                                 # add input forcing to Ti only
            P_pred    = F @ P @ FT + Q_mat                  # covariance grows with Q_mat

            # ── UPDATE 1: Ti measurement ──────────────────────────────────
            # H_Ti = [1,0,0] → S = P_pred[0,0] + R_Ti, K = P_pred[:,0] / S
            S_Ti_k = P_pred[0, 0] + R_Ti
            K_Ti   = P_pred[:, 0] / S_Ti_k                  # shape (3,)
            z_Ti_k = Ti_meas[k] - x_pred[0]                 # innovation

            x_hat = x_pred + K_Ti * z_Ti_k
            P     = P_pred - np.outer(K_Ti, P_pred[0, :])   # rank-1 downdate

            z_Ti[k] = z_Ti_k
            S_Ti[k] = S_Ti_k

            # ── UPDATE 2: CO2 measurement ──────────────────────────────────
            # H_c = [0,0,hc2] → S = hc2²*P[2,2]+R_c, K = P[:,2]*hc2/S
            c_pred_val = c[k-1] + (GV_1e6 * x_hat[2]
                                   - (qv[k] / V) * (c[k-1] - c_out)) * dt
            S_c  = hc2 * hc2 * P[2, 2] + R_c
            K_c  = P[:, 2] * (hc2 / S_c)                    # shape (3,)
            z_c  = c_meas[k] - c_pred_val

            x_hat += K_c * z_c
            P     -= np.outer(K_c, hc2 * P[2, :])           # rank-1 downdate

            # ── Store ─────────────────────────────────────────────────────
            Ti[k]    = x_hat[0]
            Tm[k]    = x_hat[1]
            N[k]     = max(0.0, x_hat[2])
            Q_int[k] = q_int_N * N[k] + q_int_const

            dc   = (GV_1e6 * N[k] - (qv[k] / V) * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        z_Ti[0]    = z_Ti[1]
        S_Ti[0]    = S_Ti[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar,
             'z_Ti': z_Ti, 'S_Ti': S_Ti}
        )
