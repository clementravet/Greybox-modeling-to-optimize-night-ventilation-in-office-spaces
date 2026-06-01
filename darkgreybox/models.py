from ast import Param
import numpy as np
import pvlib

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult



class TiTmCn2R2C_summer(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured]

    Parameters (params)
    ------------------
    Ti0, Tm0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    alpha : EMA filter parameter for occupancy update
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        alpha = params['alpha'].value 

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']              # Renamed to avoid overwriting state array

        dt = self.rec_duration   

        for k in range(1, num_rec):
            # 1) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)     # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci     # Gains
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)     # Air↔mass
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2-occupancy (c in ppm)
            #dc = (
            #    1e6 * (G / V) * N[k-1]                   # Source: ppm/h
            #    - qv[k-1] / V * (c[k-1] - c_out)         # Sink: ppm/h
            #) * dt   

            #c[k] = c[k-1] + dc

            # N from CO2 (steady-state, using updated c[k])
            #N_from_CO2 = qv[k-1] * (c[k] - c_out) / (G * 1e6)  # persons (steady-state inversion)

            # Dynamic N update: EMA filter (alpha=0.1 tunes responsiveness; adjust 0.05-0.2)
            #N[k] = (1 - alpha) * N[k-1] + alpha * N_from_CO2

            # 5) CO2-occupancy (c in ppm) with dynamic N update
            # CORRECT - uses measured CO2
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)

            # Then update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            # Finally, model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        # Set initial values for Q_int, Q_vent, Q_solar (optional: repeat first computed value)
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )
    

class TiTmCn2R2C_summer_V2(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured]
    theta_z : solar zenith angle (degrees)
    gamma_s : solar azimuth angle (degrees)

    Parameters (params)
    ------------------
    Ti0, Tm0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    alpha : EMA filter parameter for occupancy update
    gamma_g : gamma from the glazing (degrees)
    n : refractive index for IAM calculation
    K : absorption coefficient for IAM calculation (1/m)
    L : glazing thickness for IAM calculation (m)
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        alpha = params['alpha'].value 
        gamma_g = params['gamma_g'].value
        n = params['n'].value
        K = params['K'].value
        L = params['L'].value

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']              # Renamed to avoid overwriting state array
        theta_z = X['theta_z']  
        gamma_s = X['gamma_s']  

        dt = self.rec_duration  

        # Pre-compute OUTSIDE the loop
        theta_z_elevation = np.array(theta_z)  # Your current data (elevation)
        theta_z_array = 90 - theta_z_elevation  # Convert to zenith angle
        gamma_s_array = np.array(gamma_s)  # Solar azimuth angle

        # Define your surface parameters
        beta = 90  # Surface tilt: 90° for vertical wall, 0° for horizontal
        # gamma_g is your surface azimuth (already defined)

        # Manual AOI calculation using standard formula
        cos_aoi = (
            np.cos(np.radians(theta_z_array)) * np.cos(np.radians(beta)) +
            np.sin(np.radians(theta_z_array)) * np.sin(np.radians(beta)) * 
            np.cos(np.radians(gamma_s_array - gamma_g))
        )

        # Clip to valid range and set negative values (sun behind surface) to 0
        cos_correction = np.clip(cos_aoi, 0, 1)

        for k in range(1, num_rec):
            # 1) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 3) Solar gains
            #alpha_deg = np.clip(theta_z[k-1], 0, 90)  # Elevation → altitude
            #delta_gamma_deg = gamma_s[k-1] - gamma_g     
            #cos_theta = np.cos(np.radians(alpha_deg)) * np.cos(np.radians(delta_gamma_deg))
            #aoi_deg = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))  # cos_theta >=0 only!
            #iam = pvlib.iam.physical(aoi_deg, n=n, K=K, L=L)
            Q_solar[k] = g * cos_correction[k-1] * A * Ik[k-1]

            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)     # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci     # Gains
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)     # Air↔mass
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2-occupancy (c in ppm) with dynamic N update
            # CORRECT - uses measured CO2
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)

            # Then update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            # Finally, model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        # Set initial values for Q_int, Q_vent, Q_solar (optional: repeat first computed value)
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


class TiTmCn2R2C_summer_V3(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured]
    theta_z : solar zenith angle (degrees)
    gamma_s : solar azimuth angle (degrees)

    Parameters (params)
    ------------------
    Ti0, Tm0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    alpha : EMA filter parameter for occupancy update
    gamma_g : gamma from the glazing (degrees)
    n : refractive index for IAM calculation
    K : absorption coefficient for IAM calculation (1/m)
    L : glazing thickness for IAM calculation (m)
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        alpha = params['alpha'].value 
        gamma_g = params['gamma_g'].value
        n = params['n'].value
        K = params['K'].value
        L = params['L'].value

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']              # Renamed to avoid overwriting state array
        theta_z = X['theta_z']  
        gamma_s = X['gamma_s']  

        dt = self.rec_duration  

        # Pre-compute OUTSIDE the loop
        theta_z_array = 90 - np.array(theta_z)  # Solar zenith angle
        gamma_s_array = np.array(gamma_s)  # Solar azimuth angle
        # Calculate angle of incidence using pvlib (correct formula)
        aoi_deg_all = pvlib.irradiance.aoi(
            surface_tilt=90,          # 90° for vertical surface, 0° for horizontal
            surface_azimuth=gamma_g,  # Surface orientation (your gamma_g)
            solar_zenith=theta_z_array,
            solar_azimuth=gamma_s_array
        )
        # Apply physical IAM model
        iam_all = pvlib.iam.physical(aoi_deg_all, n=n, K=K, L=L)

        for k in range(1, num_rec):
            # 1) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 3) Solar gains
            #alpha_deg = np.clip(theta_z[k-1], 0, 90)  # Elevation → altitude
            #delta_gamma_deg = gamma_s[k-1] - gamma_g     
            #cos_theta = np.cos(np.radians(alpha_deg)) * np.cos(np.radians(delta_gamma_deg))
            #aoi_deg = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))  # cos_theta >=0 only!
            #iam = pvlib.iam.physical(aoi_deg, n=n, K=K, L=L)
            Q_solar[k] = g * iam_all[k-1] * A * Ik[k-1]

            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)     # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci     # Gains
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)     # Air↔mass
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2-occupancy (c in ppm) with dynamic N update
            # CORRECT - uses measured CO2
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)

            # Then update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            # Finally, model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        # Set initial values for Q_int, Q_vent, Q_solar (optional: repeat first computed value)
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


class TiTmCn2R2C_summer_V4(DarkGreyModel):
    """
    Grey-box model with Christoffer Rasmussen's time-dependent solar aperture.
    
    Key Innovation: Instead of a single constant 'g' parameter, the solar 
    aperture varies throughout the day using B-spline basis functions:
    g(t) = phi_0*B_0(t) + phi_1*B_1(t) + ... + phi_n*B_n(t)
    
    This captures how solar gains vary with sun position more accurately.
    
    Parameters (CHANGED)
    --------------------
    phi_0, phi_1, ..., phi_n : B-spline coefficients for solar aperture
    (replaces the single 'g' parameter)
    
    Inputs X (ADDED)
    ----------------
    bs_0, bs_1, ..., bs_n : Pre-computed B-spline basis functions
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters (same as before)
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        alpha = params['alpha'].value 
        gamma_g = params['gamma_g'].value
        n = params['n'].value
        K = params['K'].value
        L = params['L'].value

        # NEW: Extract B-spline coefficients instead of single 'g'
        phi_names = ['phi_a', 'phi_b', 'phi_c', 'phi_d', 'phi_e', 'phi_f', 'phi_g', 'phi_h', 'phi_i', 'phi_j']
        phi = np.array([params[name].value for name in phi_names])
        n_bsplines = len(phi)   

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']
        theta_z = X['theta_z']  
        gamma_s = X['gamma_s']

        # NEW: Extract B-spline basis functions from inputs
        bsplines = np.column_stack([X[f'bs_{i}'] for i in range(n_bsplines)])

        dt = self.rec_duration  

        # Pre-compute OUTSIDE the loop
        theta_z_array = 90 - np.array(theta_z)  # Convert elevation to zenith
        gamma_s_array = np.array(gamma_s)
        
        # Calculate angle of incidence
        aoi_deg_all = pvlib.irradiance.aoi(
            surface_tilt=90,
            surface_azimuth=gamma_g,
            solar_zenith=theta_z_array,
            solar_azimuth=gamma_s_array
        )
        
        # Apply physical IAM model
        iam_all = pvlib.iam.physical(aoi_deg_all, n=n, K=K, L=L)

        # NEW: Compute time-varying solar aperture g(t) using B-splines
        # This is Rasmussen's key innovation: g varies smoothly with time
        g_t = bsplines @ phi  # Matrix multiply: [num_rec × n] @ [n] = [num_rec]

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 3) Solar gains - UPDATED with time-varying g(t)
            # OLD: Q_solar[k] = g * iam_all[k-1] * A * Ik[k-1]
            # NEW: g is now g_t[k-1], which varies with time of day
            Q_solar[k] = g_t[k-1] * iam_all[k-1] * A * Ik[k-1]

            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2-occupancy (c in ppm) with dynamic N update
            # CORRECT - uses measured CO2
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)

            # Then update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            # Finally, model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


class TiTmCn2R2C_summer_V5(DarkGreyModel):
    """
    Grey-box model with Kalman Filter for occupancy estimation.
    
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based Kalman filter occupancy estimation


    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm) - predicted
    N   : Effective occupancy (persons) - updated via Kalman filter


    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured, used for Kalman update]


    Parameters (params)
    ------------------
    Ti0, Tm0, c0, N0 : Initial states
    P_N0 : Initial variance for N estimate
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    Q_N : Process noise variance for N (persons²)
    R_C : Measurement noise variance for CO2 (ppm²)
    """


    def model(self, params, X):
        num_rec = len(X['Ta'])


        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)
        P_N = np.zeros(num_rec)  # Variance of N estimate


        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)


        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']
        P_N[0] = params['P_N_init'].value if 'P_N_init' in params else 1.0


        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        
        # Kalman filter noise parameters
        Q_N = params['Q_N'].value if 'Q_N' in params else 0.1   # Process noise for N
        R_C = params['R_C'].value if 'R_C' in params else 100.0  # Measurement noise


        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']  # Observed CO2 for Kalman filter


        dt = self.rec_duration   


        for k in range(1, num_rec):
            # 1) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])


            # 2) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room


            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]


            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci
            ) * dt


            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt


            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm


            # 5) CO2-occupancy with KALMAN FILTER
            
            # === KALMAN FILTER FOR N ===
            # Step 1: PREDICT N (random walk model)
            N_pred = N[k-1]
            P_pred = P_N[k-1] + Q_N * dt  # Variance grows with process noise
            
            # Step 2: Compute "measured" N from observed CO2
            # Using steady-state inversion: N = (qv / (G * 10^6)) * (c_meas - c_out)
            if qv[k-1] > 0 and G > 0:
                N_meas = max(0, (qv[k-1] / (G * 1e6)) * (c_meas[k-1] - c_out))
            else:
                N_meas = 0.0
            
            # Step 3: MEASUREMENT UPDATE
            # Innovation (difference between measured and predicted N)
            y_innov = N_meas - N_pred
            
            # Innovation covariance: S = P_pred + R
            S = P_pred + R_C
            
            # Kalman gain
            K = P_pred / S if S > 1e-10 else 0.0
            
            # Update N estimate
            N[k] = N_pred + K * y_innov
            N[k] = max(0, N[k])  # N cannot be negative
            
            # Update variance
            P_N[k] = (1 - K) * P_pred


            # 6) Model CO2 for validation using updated N
            #dc = (1e6 * (G / V) * N[k] - qv[k-1] / V * (c[k-1] - c_out)) * dt
            #c[k] = c[k-1] + dc

            # Store measured CO2:
            c[k] = c_meas[k]


        # Set initial values for Q_int, Q_vent, Q_solar
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]


        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'P_N': P_N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


class TiTmCn2R2C_summer_V6(DarkGreyModel):
    """
    Grey-box model with Christoffer Rasmussen's time-dependent solar aperture.
    
    Key Innovation: Instead of a single constant 'g' parameter, the solar 
    aperture varies throughout the day using B-spline basis functions:
    g(t) = phi_0*B_0(t) + phi_1*B_1(t) + ... + phi_n*B_n(t)
    
    This captures how solar gains vary with sun position more accurately.
    
    Parameters (CHANGED)
    --------------------
    phi_0, phi_1, ..., phi_n : B-spline coefficients for solar aperture
    (replaces the single 'g' parameter)
    
    Inputs X (ADDED)
    ----------------
    bs_0, bs_1, ..., bs_n : Pre-computed B-spline basis functions
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)
        P_N = np.zeros(num_rec)  # Variance of N estimate

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']
        P_N[0] = params['P_N_init'].value if 'P_N_init' in params else 1.0

        # Parameters (same as before)
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        alpha = params['alpha'].value 
        gamma_g = params['gamma_g'].value
        n = params['n'].value
        K = params['K'].value
        L = params['L'].value
        # Kalman filter noise parameters
        Q_N = params['Q_N'].value if 'Q_N' in params else 0.1   # Process noise for N
        R_C = params['R_C'].value if 'R_C' in params else 100.0  # Measurement noise

        # NEW: Extract B-spline coefficients instead of single 'g'
        phi_names = ['phi_a', 'phi_b', 'phi_c', 'phi_d', 'phi_e', 'phi_f', 'phi_g', 'phi_h', 'phi_i', 'phi_j']
        phi = np.array([params[name].value for name in phi_names])
        n_bsplines = len(phi)   

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']
        theta_z = X['theta_z']  
        gamma_s = X['gamma_s']

        # NEW: Extract B-spline basis functions from inputs
        bsplines = np.column_stack([X[f'bs_{i}'] for i in range(n_bsplines)])

        dt = self.rec_duration  

        # Pre-compute OUTSIDE the loop
        theta_z_array = 90 - np.array(theta_z)  # Convert elevation to zenith
        gamma_s_array = np.array(gamma_s)
        
        # Calculate angle of incidence
        aoi_deg_all = pvlib.irradiance.aoi(
            surface_tilt=90,
            surface_azimuth=gamma_g,
            solar_zenith=theta_z_array,
            solar_azimuth=gamma_s_array
        )
        
        # Apply physical IAM model
        iam_all = pvlib.iam.physical(aoi_deg_all, n=n, K=K, L=L)

        # NEW: Compute time-varying solar aperture g(t) using B-splines
        # This is Rasmussen's key innovation: g varies smoothly with time
        g_t = bsplines @ phi  # Matrix multiply: [num_rec × n] @ [n] = [num_rec]

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 3) Solar gains - UPDATED with time-varying g(t)
            # OLD: Q_solar[k] = g * iam_all[k-1] * A * Ik[k-1]
            # NEW: g is now g_t[k-1], which varies with time of day
            Q_solar[k] = g_t[k-1] * iam_all[k-1] * A * Ik[k-1]

            # 4) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2-occupancy with KALMAN FILTER
            
            # === KALMAN FILTER FOR N ===
            # Step 1: PREDICT N (random walk model)
            N_pred = N[k-1]
            P_pred = P_N[k-1] + Q_N * dt  # Variance grows with process noise
            
            # Step 2: Compute "measured" N from observed CO2
            # Using steady-state inversion: N = (qv / (G * 10^6)) * (c_meas - c_out)
            if qv[k-1] > 0 and G > 0:
                N_meas = max(0, (qv[k-1] / (G * 1e6)) * (c_meas[k-1] - c_out))
            else:
                N_meas = 0.0
            
            # Step 3: MEASUREMENT UPDATE
            # Innovation (difference between measured and predicted N)
            y_innov = N_meas - N_pred
            
            # Innovation covariance: S = P_pred + R
            S = P_pred + R_C
            
            # Kalman gain
            K = P_pred / S if S > 1e-10 else 0.0
            
            # Update N estimate
            N[k] = N_pred + K * y_innov
            N[k] = max(0, N[k])  # N cannot be negative
            
            # Update variance
            P_N[k] = (1 - K) * P_pred


            # 6) Model CO2 for validation using updated N
            #dc = (1e6 * (G / V) * N[k] - qv[k-1] / V * (c[k-1] - c_out)) * dt
            #c[k] = c[k-1] + dc

            # Store measured CO2:
            c[k] = c_meas[k]

        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )

class TiTmCn2R2C_summer_V7(DarkGreyModel):
    """
    Extends V4 with inter-zone heat transfer from M neighbour rooms.
    
    New Parameters
    --------------
    Rneigh_1, ..., Rneigh_M : Thermal resistance of shared wall to each
                              neighbour room [K/W]. Can be fitted or fixed
                              from U-value * area: Rneigh_j = 1/(U_j * A_j)
    
    New Inputs X
    ------------
    T_neigh_1, ..., T_neigh_M : Interior temperature time series of each
                                 neighbour room [°C]
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate output arrays
        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_neigh = np.zeros(num_rec)  # NEW: total inter-zone heat

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters (unchanged from V4)
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        alpha         = params['alpha'].value
        g            = params['g'].value

        # NEW: Load neighbour resistances and temperature inputs dynamically
        # Detect how many neighbours are defined (Rneigh_1, Rneigh_2, ...)
        neigh_keys = sorted(
            [k for k in params if k.startswith('Rneigh_')],
            key=lambda x: int(x.split('_')[1])
        )
        M = len(neigh_keys)
        Rneigh = np.array([params[k].value for k in neigh_keys])  # shape (M,)

        # Neighbour temperature time series: shape (num_rec, M)
        T_neigh_arr = np.column_stack(
            [X[f'T_neigh_{j+1}'] for j in range(M)]
        ) if M > 0 else None

        # Inputs (unchanged)
        Ta      = X['Ta']
        Tsup    = X['Tsup']
        qv      = X['qv']
        Ik      = X['Ik']
        c_meas  = X['c']

        dt = self.rec_duration

        for k in range(1, num_rec):
            # 1) Ventilation heat (unchanged)
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains (unchanged)
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # 4) NEW: Inter-zone heat transfer from all neighbours
            # Q_neigh = sum_j [ (T_neigh_j - Ti) / Rneigh_j ]
            if M > 0:
                Q_neigh[k] = np.sum(
                    (T_neigh_arr[k-1, :] - Ti[k-1]) / Rneigh
                )

            # 5) Thermal states — only dTi changes (Q_neigh added)
            dTi = (
                (Tm[k-1]   - Ti[k-1]) / (Rim  * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_neigh[k]) / Ci
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 6) CO2 / occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN   = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc   = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_neigh[0] = Q_neigh[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent,
             'Q_solar': Q_solar, 'Q_neigh': Q_neigh}
        )

class TiTmCn2R2C_summer_V8(DarkGreyModel):
    """
    Extends TiTmCn2R2C_summer with solar gain split between Ti and Tm.

    Key change: Q_solar is now distributed between air and mass nodes
    using a fitted solar distribution factor f_sol:
      - f_sol      * Q_solar → Ti (air node, direct heating of air/light elements)
      - (1-f_sol)  * Q_solar → Tm (mass node, solar absorbed by floor/walls)

    New Parameter
    -------------
    f_sol : Solar fraction to air node Ti (0=all to mass, 1=all to air)
            Physically: ~0.3 for heavy concrete floor, ~0.6 for light construction
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate output arrays
        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters (unchanged)
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        g             = params['g'].value
        alpha         = params['alpha'].value

        # NEW: solar distribution factor
        f_sol = params['f_sol'].value  # fraction to Ti; (1 - f_sol) goes to Tm

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']

        dt = self.rec_duration

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Total solar gain (unchanged formula)
            Q_solar[k] = g * A * Ik[k-1]

            # 4) Thermal states — UPDATED: solar gain is now split
            #
            # dTi: receives f_sol fraction of solar gain
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + f_sol * Q_solar[k] + Q_int[k]) / Ci
            ) * dt

            # dTm: receives (1 - f_sol) fraction of solar gain
            #      (previously had no direct solar input)
            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
                + (1.0 - f_sol) * Q_solar[k] / Cm
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) CO2 / occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN   = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc   = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


class TiTmCn2R2C_summer_V9(DarkGreyModel):
    """
    Grey-box model combining:
      1. Rasmussen's time-dependent solar aperture via B-splines (from V4)
      2. Solar gain split between air node Ti and mass node Tm (NEW)
      3. Inter-zone heat transfer from M neighbour rooms (NEW)

    Equations
    ---------
    Ci * dTi/dt = (Tm-Ti)/Rim + (Ta-Ti)/Rout
                  + Q_vent + Q_int
                  + f_sol * Q_solar
                  + sum_j[ (T_neigh_j - Ti) / Rneigh_j ]

    Cm * dTm/dt = (Ti-Tm)/Rim
                  + (1 - f_sol) * Q_solar

    New Parameters (vs V4)
    ----------------------
    f_sol      : Solar fraction absorbed by air node Ti [0,1]
                 (1-f_sol) goes to mass node Tm
                 Init: 0.4, bounds: [0, 1]

    Rneigh_1, Rneigh_2, ... : Thermal resistance to each neighbour room [K/W]
                 Compute from geometry: Rneigh_j = 1 / (U_j * A_j)
                 e.g. U=1.8 W/m²K, A=12m² -> Rneigh = 0.046 K/W
                 Init: ~0.05, bounds: [0.005, 2.0]

    New Inputs X (vs V4)
    --------------------
    T_neigh_1, T_neigh_2, ... : Interior temperature of each neighbour [°C]
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti = np.zeros(num_rec)
        Tm = np.zeros(num_rec)
        c  = np.zeros(num_rec)
        N  = np.zeros(num_rec)

        # Allocate output arrays
        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_neigh = np.zeros(num_rec)  # total inter-zone heat flux [W]

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # ── Unchanged parameters ──────────────────────────────────────────
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        alpha         = params['alpha'].value
        gamma_g       = params['gamma_g'].value
        n             = params['n'].value
        K             = params['K'].value
        L             = params['L'].value

        # ── NEW: solar split parameter ────────────────────────────────────
        f_sol = params['f_sol'].value   # fraction of Q_solar to Ti

        # ── B-spline coefficients (unchanged from V4) ─────────────────────
        phi_names  = ['phi_a', 'phi_b', 'phi_c', 'phi_d', 'phi_e',
                      'phi_f', 'phi_g', 'phi_h', 'phi_i', 'phi_j']
        phi        = np.array([params[name].value for name in phi_names])
        n_bsplines = len(phi)

        # ── NEW: neighbour resistances (auto-detect from params) ──────────
        neigh_keys = sorted(
            [k for k in params if k.startswith('Rneigh_')],
            key=lambda x: int(x.split('_')[1])
        )
        M      = len(neigh_keys)
        Rneigh = np.array([params[k].value for k in neigh_keys])  # shape (M,)

        # Neighbour temperature time series: shape (num_rec, M)
        T_neigh_arr = (
            np.column_stack([X[f'T_neigh_{j+1}'] for j in range(M)])
            if M > 0 else None
        )

        # ── Inputs (unchanged from V4) ────────────────────────────────────
        Ta      = X['Ta']
        Tsup    = X['Tsup']
        qv      = X['qv']
        Ik      = X['Ik']
        c_meas  = X['c']
        theta_z = X['theta_z']
        gamma_s = X['gamma_s']

        # B-spline basis functions: shape (num_rec, n_bsplines)
        bsplines = np.column_stack([X[f'bs_{i}'] for i in range(n_bsplines)])

        dt = self.rec_duration

        # ── Pre-compute AOI and IAM outside loop (unchanged) ─────────────
        theta_z_array = 90 - np.array(theta_z)
        gamma_s_array = np.array(gamma_s)

        aoi_deg_all = pvlib.irradiance.aoi(
            surface_tilt=90,
            surface_azimuth=gamma_g,
            solar_zenith=theta_z_array,
            solar_azimuth=gamma_s_array
        )
        iam_all = pvlib.iam.physical(aoi_deg_all, n=n, K=K, L=L)

        # Time-varying solar aperture: shape (num_rec,)
        g_t = bsplines @ phi

        # ── Main simulation loop ──────────────────────────────────────────
        for k in range(1, num_rec):

            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Total solar gain (B-spline aperture + IAM, unchanged)
            Q_solar[k] = g_t[k-1] * iam_all[k-1] * A * Ik[k-1]

            # 4) NEW: inter-zone heat from all neighbour rooms → Ti only
            if M > 0:
                Q_neigh[k] = np.sum(
                    (T_neigh_arr[k-1, :] - Ti[k-1]) / Rneigh
                )

            # 5) Thermal states
            #
            # Ti receives: f_sol * Q_solar  +  full Q_neigh
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim  * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_int[k] + f_sol * Q_solar[k] + Q_neigh[k]) / Ci
            ) * dt

            # Tm receives: (1 - f_sol) * Q_solar  (no neighbour term)
            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
                + (1.0 - f_sol) * Q_solar[k] / Cm
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 6) CO2 / occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN   = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc   = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_neigh[0] = Q_neigh[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent,
             'Q_solar': Q_solar, 'Q_neigh': Q_neigh}
        )
    

class TiTmCn2R2C_summer_V11(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)
      - Kalman Filter for optimal N estimation (replaces EMA / alpha)

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)  ← estimated via Kalman Filter

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured]

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
    Met               : Metabolic rate (Met), fixed=1.2 for light office work
    c_out             : Outdoor CO2 fraction (ppm)
    alpha_lat         : Latent heat per person at 1 Met (W/person), fitted ~35-45
    q_equip_var       : Equipment heat gain per person (W/person)
    q_equip_const     : Constant equipment gains (W/m²)
    g                 : Total solar energy transmittance of the glazing
    sigma_N           : Process noise std dev for N [persons] — replaces alpha
    sigma_c           : CO2 measurement noise std dev [ppm]   — from sensor spec
    P0_N              : Initial state variance for N [persons²]
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters (unchanged)
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

        # Met-dependent parameters
        Met       = params['Met'].value
        G_base    = params['G_base'].value
        alpha_lat = params['alpha_lat'].value

        # ── DERIVED ───────────────────────────────────────────────────────
        G      = G_base * Met
        q_sens = Met * (100.0 - alpha_lat)

        # ── Kalman Filter parameters ──────────────────────────────────────
        sigma_N = params['sigma_N'].value   # process noise [persons/step]
        sigma_c = params['sigma_c'].value   # CO2 measurement noise [ppm]
        Q_kf    = sigma_N ** 2              # process noise variance
        R_kf    = sigma_c ** 2              # measurement noise variance

        # KF initial state
        N_hat = float(params['N0'])
        P     = params['P0_N'].value

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']

        dt = self.rec_duration

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_sens + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # 4) Thermal states (unchanged)
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k]) / Ci
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 5) Kalman Filter for N ─────────────────────────────────────

            # PREDICT: random walk — no expected change, variance grows
            N_hat_pred = N_hat
            P_pred     = P + Q_kf

            # Predicted CO2 at N_hat_pred via CO2 mass balance
            c_pred = c[k-1] + (
                (G / V) * 1e6 * N_hat_pred
                - (qv[k] / V) * (c[k-1] - c_out)
            ) * dt

            # Measurement Jacobian: H = d(c_pred)/d(N) = (G/V)*1e6*dt
            H = (G / V) * 1e6 * dt

            # UPDATE: correct N using real vs predicted CO2
            z    = c_meas[k] - c_pred          # innovation
            S_kf = H * P_pred * H + R_kf       # innovation covariance
            K    = P_pred * H / S_kf           # Kalman gain

            N_hat = N_hat_pred + K * z         # corrected estimate
            P     = (1.0 - K * H) * P_pred    # updated variance

            N[k] = max(0.0, N_hat)             # physical constraint

            # 6) CO2 state update using KF-estimated N (unchanged structure)
            dc   = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar}
        )


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
        cp_rho_3600 = rho_air * cp_air / 3600.0  # rho*cp/3600
        gA          = g * A                       # solar gain factor
        GV_1e6      = (G / V) * 1e6              # CO2 emission factor
        q_int_N     = q_sens + q_equip_var        # heat gain per person
        q_int_const = q_equip_const * S           # constant heat gain

        # ── F is constant (depends only on params, not on k) ─────────────
        # Precompute ONCE — avoids rebuilding a 3x3 array every iteration
        F = np.array([
            [1.0 - dt/(Rim*Ci) - dt/(Rout*Ci),
                   dt/(Rim*Ci),
                   q_int_N * dt / Ci],
            [dt/(Rim*Cm),
             1.0 - dt/(Rim*Cm),
             0.0],
            [0.0, 0.0, 1.0]
        ])
        FT = F.T   # precompute transpose too

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
            x_pred[0] += b0                    # add input forcing to Ti only
            P_pred    = F @ P @ FT + Q_mat     # covariance grows with Q_mat

            # ── UPDATE 1: Ti measurement ──────────────────────────────────
            # H_Ti = [1,0,0] → S = P_pred[0,0] + R_Ti, K = P_pred[:,0] / S
            S_Ti_k = P_pred[0, 0] + R_Ti
            K_Ti   = P_pred[:, 0] / S_Ti_k        # shape (3,)
            z_Ti_k = Ti_meas[k] - x_pred[0]       # innovation

            x_hat = x_pred + K_Ti * z_Ti_k
            P     = P_pred - np.outer(K_Ti, P_pred[0, :])   # rank-1 downdate

            z_Ti[k] = z_Ti_k
            S_Ti[k] = S_Ti_k

            # ── UPDATE 2: CO2 measurement ──────────────────────────────────
            # H_c = [0,0,hc2] → S = hc2²*P[2,2]+R_c, K = P[:,2]*hc2/S
            c_pred_val = c[k-1] + (GV_1e6 * x_hat[2]
                                   - (qv[k] / V) * (c[k-1] - c_out)) * dt
            S_c  = hc2 * hc2 * P[2, 2] + R_c
            K_c  = P[:, 2] * (hc2 / S_c)          # shape (3,)
            z_c  = c_meas[k] - c_pred_val

            x_hat += K_c * z_c
            P     -= np.outer(K_c, hc2 * P[2, :]) # rank-1 downdate

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


class TiTmTfCn3R3C_summer_V13(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R3C thermal model (Ti, Tm, Tf)
          Ti : indoor air temperature
          Tm : slow structural mass (walls, slab)
          Tf : fast interior mass (furniture, ceiling, fittings)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model
      - Full multivariate Kalman Filter on [Ti, Tm, Tf, N]

    States:  Ti, Tm, Tf, c, N
    Inputs:  Ta, Tsup, qv, Ik, c (CO2 meas), Ti_meas
    New vs V12: Rf, Cf, sigma_Tf, Tf0, P0_Tf
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # ── Allocate ──────────────────────────────────────────────────────
        Ti      = np.zeros(num_rec)
        Tm      = np.zeros(num_rec)
        Tf      = np.zeros(num_rec)
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
        Tf[0] = params['Tf0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # ── Physical parameters ───────────────────────────────────────────
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        Rf            = params['Rf'].value          # ← NEW: fast mass resistance
        Cf            = params['Cf'].value          # ← NEW: fast mass capacitance
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
        sigma_Tf      = params['sigma_Tf'].value    # ← NEW
        sigma_N       = params['sigma_N'].value
        sigma_Ti_meas = params['sigma_Ti_meas'].value
        sigma_c       = params['sigma_c'].value

        # Process noise covariance (4x4 diagonal)
        Q_mat = np.diag([sigma_Ti**2, sigma_Tm**2, sigma_Tf**2, sigma_N**2])

        # ── KF initial state and covariance (4D) ─────────────────────────
        x_hat = np.array([Ti[0], Tm[0], Tf[0], float(params['N0'])])
        P     = np.diag([params['P0_Ti'].value,
                         params['P0_Tm'].value,
                         params['P0_Tf'].value,     # ← NEW
                         params['P0_N'].value])
        I4    = np.eye(4)

        # ── Inputs → numpy ────────────────────────────────────────────────
        Ta      = np.asarray(X['Ta'],      dtype=float)
        Tsup    = np.asarray(X['Tsup'],    dtype=float)
        qv      = np.asarray(X['qv'],      dtype=float)
        Ik      = np.asarray(X['Ik'],      dtype=float)
        c_meas  = np.asarray(X['c'],       dtype=float)
        Ti_meas = np.asarray(X['Ti_meas'], dtype=float)

        dt = self.rec_duration

        # ── Precompute scalars ────────────────────────────────────────────
        cp_rho_3600 = rho_air * cp_air / 3600.0
        gA          = g * A
        GV_1e6      = (G / V) * 1e6
        q_int_N     = q_sens + q_equip_var
        q_int_const = q_equip_const * S

        # ── F is constant — precompute once ──────────────────────────────
        # State order: [Ti, Tm, Tf, N]
        # Ti gains flux from Tm (via Rim), Ta (via Rout), AND Tf (via Rf)
        # Tf gains flux from Ti only (via Rf)
        F = np.array([
            [1.0 - dt/(Rim*Ci) - dt/(Rout*Ci) - dt/(Rf*Ci),
                   dt/(Rim*Ci),
                   dt/(Rf*Ci),
                   q_int_N * dt / Ci],
            [dt/(Rim*Cm),
             1.0 - dt/(Rim*Cm),
             0.0,
             0.0],
            [dt/(Rf*Cf),
             0.0,
             1.0 - dt/(Rf*Cf),
             0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        FT = F.T

        # ── Measurement constants ─────────────────────────────────────────
        # H_Ti  = [1, 0, 0, 0] → K = P[:,0] / (P[0,0] + R_Ti)
        # H_c   = [0, 0, 0, hc3] → K = P[:,3]*hc3 / (hc3²*P[3,3] + R_c)
        R_Ti = sigma_Ti_meas ** 2
        hc3  = GV_1e6 * dt          # H_c non-zero element (now index 3)
        R_c  = sigma_c ** 2

        # ── Main loop ─────────────────────────────────────────────────────
        for k in range(1, num_rec):

            Q_vent_k  = cp_rho_3600 * qv[k-1] * (Tsup[k-1] - x_hat[0])
            Q_solar_k = gA * Ik[k-1]
            Q_vent[k]  = Q_vent_k
            Q_solar[k] = Q_solar_k

            # Forcing: only Ti (index 0) receives external inputs
            b0 = (Q_vent_k + Q_solar_k + q_int_const) * dt / Ci

            # ── PREDICT ───────────────────────────────────────────────────
            x_pred    = F @ x_hat
            x_pred[0] += b0
            P_pred    = F @ P @ FT + Q_mat

            # ── UPDATE 1: Ti measurement — H=[1,0,0,0] ───────────────────
            S_Ti_k = P_pred[0, 0] + R_Ti
            K_Ti   = P_pred[:, 0] / S_Ti_k
            z_Ti_k = Ti_meas[k] - x_pred[0]

            x_hat = x_pred + K_Ti * z_Ti_k
            P     = P_pred - np.outer(K_Ti, P_pred[0, :])

            z_Ti[k] = z_Ti_k
            S_Ti[k] = S_Ti_k

            # ── UPDATE 2: CO2 measurement — H=[0,0,0,hc3] ────────────────
            c_pred_val = c[k-1] + (GV_1e6 * x_hat[3]
                                   - (qv[k] / V) * (c[k-1] - c_out)) * dt
            S_c  = hc3 * hc3 * P[3, 3] + R_c
            K_c  = P[:, 3] * (hc3 / S_c)
            z_c  = c_meas[k] - c_pred_val

            x_hat += K_c * z_c
            P     -= np.outer(K_c, hc3 * P[3, :])

            # ── Store ─────────────────────────────────────────────────────
            Ti[k]    = x_hat[0]
            Tm[k]    = x_hat[1]
            Tf[k]    = x_hat[2]
            N[k]     = max(0.0, x_hat[3])
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
            {'Ti': Ti, 'Tm': Tm, 'Tf': Tf, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar,
             'z_Ti': z_Ti, 'S_Ti': S_Ti}
        )



class TiTmxvCn2R2C_winter_V2(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Direct radiator heating (no valve/radiator state)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)


    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)


    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    Tfor    : Supply water temperature to radiator circuit (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    MVV     : Heating command (%) for radiator
    c       : CO2 concentration (ppm) [measured]


    Parameters (params)
    ------------------
    Ti0, Tm0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    alpha : EMA filter parameter for occupancy update
    alpha_rad : Radiator effectiveness [W/K] (combines all heating effects)
    """


    def model(self, params, X):
        num_rec = len(X['Ta'])


        # Allocate states (ONLY 4 states: Ti, Tm, c, N)
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)


        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_heat = np.zeros(num_rec)
        #Q_inf = np.zeros(num_rec)


        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']


        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        alpha = params['alpha'].value 
        alpha_rad = params['alpha_rad'].value
        #q_inf = params['q_inf'].value  # Infiltration rate (air changes per hour)


        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        Tfor   = X['Tfor']
        qv     = X['qv']
        Ik     = X['Ik']
        MVV    = X['MVV']
        c_meas = X['c']


        dt = self.rec_duration   


        for k in range(1, num_rec):
            # 1) Radiator heat - direct calculation (no state dynamics)
            Q_heat[k] = alpha_rad * (MVV[k-1]/100) * (Tfor[k-1] - Ti[k-1])
            #Q_heat[k] = alpha_rad * (MVV[k-1]/100) * 35.0
            #MVV_eff = 0.1 + 0.9*(MVV[k-1]/100)  # 10% minimum even when "off"
            #Q_heat[k] = alpha_rad * MVV_eff * (Tfor[k-1] - Ti[k-1])

            # Infiltration:
            #Q_inf[k] = q_inf * rho_air * cp_air * (V/3600) * (Ta[k-1] - Ti[k-1])


            # 2) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])


            # 3) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room


            # 4) Solar gains
            Q_solar[k] = g * A * Ik[k-1]


            # 5) Thermal states
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)       # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k]) / Ci  # All gains
            ) * dt


            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)       # Air↔mass
            ) * dt


            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm


            # 6) CO2-occupancy (c in ppm) with dynamic N update
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)


            # Update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)


            # Model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc


        # Set initial values for outputs
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_heat[0] = Q_heat[1]


        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar, 'Q_heat': Q_heat}
        )


class TiTmThPhiCn2R2C_winter_V3(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm) + radiator dynamics (T_h, Φ)
      - Radiator with thermal mass and water flow dynamics
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)
    
    Based on: https://www.sciencedirect.com/science/article/pii/S0378778821007416

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    T_h : Radiator surface temperature (°C)
    Phi : Water flow rate through radiator (kg/s)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    Tfor    : Supply water temperature to radiator circuit (°C)
    qv      : Ventilation flow rate (m3/h)
    Ik      : Irradiance (W/m²)
    MVV     : Heating command (%) for radiator
    c       : CO2 concentration (ppm) [measured]

    Parameters (params)
    ------------------
    Ti0, Tm0, Th0, Phi0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Ch  : Radiator heat capacity (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    Rrh : Resistance radiator-air (K/W)
    Phi_max : Maximum water flow rate (kg/s)
    Cf  : Flow time constant (s)
    cp_w : Specific heat of water (J/kg·K)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction (ppm)
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    alpha : EMA filter parameter for occupancy update
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states (6 states: Ti, Tm, Th, Phi, c, N)
        Ti  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        Th  = np.zeros(num_rec)  # Radiator temperature
        Phi = np.zeros(num_rec)  # Water flow rate
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Allocate arrays for outputs
        Q_int = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_heat = np.zeros(num_rec)  # Heat from radiator to air
        Q_water = np.zeros(num_rec)  # Heat from water to radiator

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        Th[0] = params['Th0']
        Phi[0] = params['Phi0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Ch   = params['Ch'].value
        Rim  = params['Rim'].value
        Rout = params['Rout'].value
        Rrh  = params['Rrh'].value  # Radiator-air resistance
        Phi_max = params['Phi_max'].value
        Cf   = params['Cf'].value
        cp_w = params['cp_w'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V = params['V'].value
        S = params['S'].value
        A = params['A'].value
        G = params['G'].value
        c_out = params['c_out'].value
        rho_air = params['rho_air'].value
        cp_air = params['cp_air'].value
        g = params['g'].value
        alpha = params['alpha'].value

        # Inputs
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        Tfor   = X['Tfor']
        qv     = X['qv']
        Ik     = X['Ik']
        MVV    = X['MVV']
        c_meas = X['c']

        dt = self.rec_duration

        for k in range(1, num_rec):
            # 1) Water flow dynamics: dΦ/dt = (1/Cf) * (Φ_max * MVV - Φ)
            dPhi = (1.0 / Cf) * (Phi_max * (MVV[k-1]/100) - Phi[k-1]) * dt
            Phi[k] = max(0, Phi[k-1] + dPhi)  # Ensure non-negative flow

            # 2) Heat from water to radiator: Q_water = Φ * cp_w * (Tfor - Th)
            Q_water[k] = Phi[k-1] * cp_w * (Tfor[k-1] - Th[k-1])

            # 3) Heat from radiator to air: Q_heat = (Th - Ti) / Rrh
            Q_heat[k] = (Th[k-1] - Ti[k-1]) / Rrh

            # 4) Radiator temperature dynamics: dTh/dt = (1/Ch) * (Q_water - Q_heat)
            dTh = (Q_water[k] - Q_heat[k]) / Ch * dt
            Th[k] = Th[k-1] + dTh

            # 5) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent[k] = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 6) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int_room = q_equip_const * S
            Q_int[k] = Q_int_occ + Q_int_room

            # 7) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # 8) Indoor air temperature dynamics
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)       # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k]) / Ci  # All gains
            ) * dt

            Ti[k] = Ti[k-1] + dTi

            # 9) Thermal mass temperature dynamics
            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)       # Air↔mass
            ) * dt

            Tm[k] = Tm[k-1] + dTm

            # 10) CO2-occupancy (c in ppm) with dynamic N update
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)

            # Update N state
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            # Model c for validation
            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        # Set initial values for outputs
        Q_int[0] = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_heat[0] = Q_heat[1]
        Q_water[0] = Q_water[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'Th': Th, 'Phi': Phi, 'c': c, 'N': N, 
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar, 
             'Q_heat': Q_heat, 'Q_water': Q_water}
        )


class TiTmCn2R2C_winter_V4(DarkGreyModel):
    def model(self, params, X):
        num_rec = len(X['Ta'])

        Ti    = np.zeros(num_rec)
        Tm    = np.zeros(num_rec)
        c     = np.zeros(num_rec)
        N     = np.zeros(num_rec)
        Tret  = np.zeros(num_rec)

        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_heat  = np.zeros(num_rec)

        Ti[0]   = params['Ti0']
        Tm[0]   = params['Tm0']
        c[0]    = params['c0']
        N[0]    = params['N0']
        Tret[0] = params['Tret0']

        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        g             = params['g'].value
        alpha         = params['alpha'].value
        K             = params['K'].value
        Q_flow_max    = params['Q_flow_max'].value
        R_valve       = params['R_valve'].value
        cp_water      = params['cp_water'].value
        rho_water     = params['rho_water'].value

        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']
        MVV    = X['MVV']
        Tfor   = X['Tfor']

        dt = self.rec_duration

        eps_lmtd   = 0.1   # CHANGE 1
        max_dT_ret = 2.0   # CHANGE 2

        # CHANGE 3 (updated): smooth ramp threshold for equal-percentage valve
        #   Below MVV_min, flow is linearly ramped to 0 rather than hard-cut.
        #   This keeps the equal-percentage law intact above MVV_min while
        #   eliminating the discontinuous jump that caused Q_heat spikes.
        MVV_min = 0.02  # tune if your valve data shows a different dead-band

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # ----------------------------------------------------------------
            # 4) RADIATOR HEAT TRANSFER
            # ----------------------------------------------------------------

            # 4a) Mass flow — equal-percentage + smooth ramp
            MVV_k = np.clip(MVV[k-1], 0.0, 1.0)
            if MVV_k < 1e-6:
                m_dot = 0.0
            else:
                Q_flow_ep = Q_flow_max * (R_valve ** (MVV_k - 1))
                smooth    = min(MVV_k / MVV_min, 1.0)
                Q_flow    = Q_flow_ep * smooth
                m_dot     = rho_water * (Q_flow / 3600)

            # 4b) LMTD + Q_heat — zero when valve closed, LMTD when open
            if m_dot < 1e-6:
                # Valve closed: no flow, no heat transfer
                Q_heat[k] = 0.0
                LMTD      = 0.0
            else:
                dT_supply = Tfor[k-1] - Ti[k-1]
                dT_return = Tret[k-1] - Ti[k-1]

                dT_supply_eff = max(dT_supply, eps_lmtd)
                dT_return_eff = np.clip(dT_return, eps_lmtd, dT_supply_eff - eps_lmtd)

                if abs(dT_supply_eff - dT_return_eff) < 1e-6:
                    LMTD = dT_supply_eff  # L'Hopital guard
                else:
                    LMTD = (dT_supply_eff - dT_return_eff) / np.log(dT_supply_eff / dT_return_eff)

                Q_lmtd = K * (LMTD ** 1.3)

                # Flow-limited cap
                Tret_prev_eff = min(Tret[k-1], Tfor[k-1] - eps_lmtd)
                Q_flow_limit  = m_dot * cp_water * (Tfor[k-1] - Tret_prev_eff)
                Q_heat[k]     = min(Q_lmtd, Q_flow_limit)

            # 4c) T_ret dynamics
            if m_dot > 1e-6:
                T_ret_target = Tfor[k-1] - Q_heat[k] / (m_dot * cp_water)
                tau_ret      = 300.0
                dT_ret       = ((T_ret_target - Tret[k-1]) / tau_ret) * dt
            else:
                # Valve closed: T_ret decays toward Ti
                dT_ret = ((Ti[k-1] - Tret[k-1]) / 300.0) * dt

            dT_ret  = np.clip(dT_ret, -max_dT_ret, max_dT_ret)
            Tret[k] = Tret[k-1] + dT_ret
            Tret[k] = np.clip(Tret[k], Ti[k-1] + eps_lmtd, Tfor[k-1] - eps_lmtd)



            # ----------------------------------------------------------------
            # 5) Thermal states (unchanged)
            # ----------------------------------------------------------------
            dTi = (
                (Tm[k-1]  - Ti[k-1]) / (Rim  * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k]) / Ci
            ) * dt

            dTm = ((Ti[k-1] - Tm[k-1]) / (Rim * Cm)) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 6) CO2-occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_heat[0]  = Q_heat[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Tret': Tret,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar, 'Q_heat': Q_heat}
        )



class TiTmCn2R2C_winter_V5(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm) + radiator heat transfer
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in ppm)
      - LMTD-based radiator model with black-box valve flow

    States / Inputs / Parameters unchanged — see original docstring.
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        Ti    = np.zeros(num_rec)
        Tm    = np.zeros(num_rec)
        c     = np.zeros(num_rec)
        N     = np.zeros(num_rec)
        Tret  = np.zeros(num_rec)

        Q_int   = np.zeros(num_rec)
        Q_vent  = np.zeros(num_rec)
        Q_solar = np.zeros(num_rec)
        Q_heat  = np.zeros(num_rec)

        Ti[0]   = params['Ti0']
        Tm[0]   = params['Tm0']
        c[0]    = params['c0']
        N[0]    = params['N0']
        Tret[0] = params['Tret0']

        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        g             = params['g'].value
        alpha         = params['alpha'].value

        K         = params['K'].value
        K_flow    = params['K_flow'].value
        n_valve   = params['n_valve'].value
        cp_water  = params['cp_water'].value
        rho_water = params['rho_water'].value

        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']
        MVV    = X['MVV']
        Tfor   = X['Tfor']

        dt = self.rec_duration

        eps_lmtd   = 0.1  # soft floor for LMTD deltas
        max_dT_ret = 2.0  # max T_ret change per timestep

        for k in range(1, num_rec):
            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # ----------------------------------------------------------------
            # 4) RADIATOR HEAT TRANSFER
            # ----------------------------------------------------------------

            # 4a) Mass flow — black-box power law (unchanged logic, cleaner threshold)
            MVV_k = np.clip(MVV[k-1], 0.0, 1.0)
            if MVV_k < 1e-6:
                m_dot = 0.0
            else:
                m_dot = K_flow * (MVV_k ** n_valve)  # kg/s

            # 4b) LMTD + Q_heat — zero when valve closed, LMTD when open
            if m_dot < 1e-6:
                # Valve closed: Q_heat = 0, T_ret decays in 4c
                Q_heat[k] = 0.0
                LMTD      = 0.0
            else:
                dT_supply = Tfor[k-1] - Ti[k-1]
                dT_return = Tret[k-1] - Ti[k-1]

                dT_supply_eff = max(dT_supply, eps_lmtd)
                dT_return_eff = np.clip(dT_return, eps_lmtd, dT_supply_eff - eps_lmtd)

                # L'Hopital guard: when supply ≈ return, LMTD → dT arithmetically
                if abs(dT_supply_eff - dT_return_eff) < 1e-6:
                    LMTD = dT_supply_eff
                else:
                    LMTD = (dT_supply_eff - dT_return_eff) / np.log(dT_supply_eff / dT_return_eff)

                Q_lmtd = K * (LMTD ** 1.3)

                # Flow-limited cap
                Tret_prev_eff = min(Tret[k-1], Tfor[k-1] - eps_lmtd)
                Q_flow_limit  = m_dot * cp_water * (Tfor[k-1] - Tret_prev_eff)
                Q_heat[k]     = min(Q_lmtd, Q_flow_limit)

            # 4c) T_ret dynamics
            if m_dot > 1e-6:
                T_ret_target = Tfor[k-1] - Q_heat[k] / (m_dot * cp_water)
                tau_ret      = 300.0
                dT_ret       = ((T_ret_target - Tret[k-1]) / tau_ret) * dt
            else:
                # Valve closed: T_ret decays toward Ti
                dT_ret = ((Ti[k-1] - Tret[k-1]) / 300.0) * dt

            dT_ret  = np.clip(dT_ret, -max_dT_ret, max_dT_ret)
            Tret[k] = Tret[k-1] + dT_ret
            Tret[k] = np.clip(Tret[k], Ti[k-1] + eps_lmtd, Tfor[k-1] - eps_lmtd)

            # ----------------------------------------------------------------
            # 5) Thermal states (unchanged)
            # ----------------------------------------------------------------
            dTi = (
                (Tm[k-1]  - Ti[k-1]) / (Rim  * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k]) / Ci
            ) * dt

            dTm = ((Ti[k-1] - Tm[k-1]) / (Rim * Cm)) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 6) CO2-occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        Q_int[0]   = Q_int[1]
        Q_vent[0]  = Q_vent[1]
        Q_solar[0] = Q_solar[1]
        Q_heat[0]  = Q_heat[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N, 'Tret': Tret,
             'Q_int': Q_int, 'Q_vent': Q_vent, 'Q_solar': Q_solar, 'Q_heat': Q_heat}
        )


class TiTmCn2R2C_winter_V6(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R2C thermal model (Ti, Tm)
      - Ventilation, solar, internal gains (unchanged from summer)
      - ε-NTU radiator model replacing LMTD
        → Handles MVV=0 gracefully, no T_ret state needed

    States
    ------
    Ti  : Indoor air temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    c   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)
    ### REMOVED: T_ret (no longer needed as a state)

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m³/h)
    Ik      : Irradiance (W/m²)
    c       : CO2 concentration (ppm) [measured]
    MVV     : Radiator valve position (0–1)
    Tfor   : Hot water supply temperature to radiator (°C)

    Parameters
    ----------
    Ti0, Tm0, c0, N0  : Initial states (T_ret0 REMOVED)
    Ci, Cm            : Capacitances (J/K)
    Rim, Rout         : Thermal resistances (K/W)
    rho_air, cp_air   : Air properties
    V, S, A           : Room geometry
    G, c_out          : CO2 parameters
    q_pers, q_equip_var, q_equip_const : Gains
    g, alpha          : Solar transmittance, EMA filter
    UA        : Radiator heat transfer coeff (W/K)  ← replaces K
    Q_flow_max: Max water flow at full opening (l/h) ← estimate
    R_valve   : Valve rangeability (-)              ← fix at 30 or estimate
    cp_water  : 4186 J/(kg·K)                       ← fix
    rho_water : 1000 kg/m³                          ← fix
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # --- Allocate states ---
        Ti = np.zeros(num_rec)
        Tm = np.zeros(num_rec)
        c  = np.zeros(num_rec)
        N  = np.zeros(num_rec)

        # --- Allocate outputs ---
        Q_int  = np.zeros(num_rec)
        Q_vent = np.zeros(num_rec)
        Q_solar= np.zeros(num_rec)
        Q_heat = np.zeros(num_rec)

        # --- Initial conditions ---
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']
        c[0]  = params['c0']
        N[0]  = params['N0']

        # --- Parameters (unchanged from summer) ---
        Ci            = params['Ci'].value
        Cm            = params['Cm'].value
        Rim           = params['Rim'].value
        Rout          = params['Rout'].value
        q_pers        = params['q_pers'].value
        q_equip_var   = params['q_equip_var'].value
        q_equip_const = params['q_equip_const'].value
        V             = params['V'].value
        S             = params['S'].value
        A             = params['A'].value
        G             = params['G'].value
        c_out         = params['c_out'].value
        rho_air       = params['rho_air'].value
        cp_air        = params['cp_air'].value
        g             = params['g'].value
        alpha         = params['alpha'].value

        # --- NEW heating parameters ---
        UA          = params['UA'].value        # Radiator heat transfer (W/K)
        Q_flow_max  = params['Q_flow_max'].value  # Max flow (l/h)
        R_valve     = params['R_valve'].value   # Rangeability, e.g. 30
        cp_water    = params['cp_water'].value  # 4186 J/kgK
        rho_water   = params['rho_water'].value # 1000 kg/m³

        # --- Inputs ---
        Ta     = X['Ta']
        Tsup   = X['Tsup']
        qv     = X['qv']
        Ik     = X['Ik']
        c_meas = X['c']
        MVV    = np.array(X['MVV'])/100.0    # Radiator valve position (0–1), convert from % to fraction
        Tfor  = X['Tfor']  # Hot water supply temperature

        dt = self.rec_duration

        for k in range(1, num_rec):

            # 1) Ventilation heat
            Q_vent[k] = rho_air * cp_air * (qv[k-1] / 3600) * (Tsup[k-1] - Ti[k-1])

            # 2) Internal gains
            Q_int[k] = (q_pers + q_equip_var) * N[k-1] + q_equip_const * S

            # 3) Solar gains
            Q_solar[k] = g * A * Ik[k-1]

            # 4) ε-NTU radiator heat transfer
            # Ref: Wetter et al. (2014), Modelica Buildings Library
            #      Bouskela & El Hefni (2014), Modelica Conference

            # 4a) Mass flow rate: equal percentage valve (Option 1)
            if MVV[k-1] > 0.05:
                Q_flow = Q_flow_max * (R_valve ** (MVV[k-1] - 1))  # l/h
                m_dot  = rho_water * (Q_flow / 3600)               # kg/s
            else:
                m_dot = 0.0

            # 4b) ε-NTU: handles m_dot=0 naturally
            # Regularize to avoid division by zero (Bouskela & El Hefni, 2014)
            m_dot_safe = max(m_dot, 1e-6)               # tiny regularization
            C_water    = m_dot_safe * cp_water           # W/K
            NTU        = UA / C_water                    # dimensionless
            epsilon    = 1.0 - np.exp(-NTU)             # effectiveness (0–1)

            # 4c) Heat transfer to room
            dT_available = max(0.0, Tfor[k-1] - Ti[k-1])
            Q_heat[k] = epsilon * C_water * dT_available
            # When m_dot=0: C_water→0, epsilon→1, product→0 ✓
            # When m_dot large: epsilon→(1-e^-NTU), Q_heat→UA*LMTD ✓

            # 5) Thermal states (unchanged structure, Q_heat added)
            dTi = (
                (Tm[k-1] - Ti[k-1]) / (Rim * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k]) / Ci
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Tm[k] = Tm[k-1] + dTm

            # 6) CO2-occupancy (unchanged)
            N_from_CO2 = (qv[k] / (G * 1e6)) * (c_meas[k] - c_out)
            dN = (alpha / dt) * (N_from_CO2 - N[k-1]) * dt
            N[k] = max(0, N[k-1] + dN)

            dc = (1e6 * (G / V) * N[k] - qv[k] / V * (c[k-1] - c_out)) * dt
            c[k] = c[k-1] + dc

        # Fill t=0
        Q_int[0]  = Q_int[1]
        Q_vent[0] = Q_vent[1]
        Q_solar[0]= Q_solar[1]
        Q_heat[0] = Q_heat[1]

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Tm': Tm, 'c': c, 'N': N,
             'Q_int': Q_int, 'Q_vent': Q_vent,
             'Q_solar': Q_solar, 'Q_heat': Q_heat}
        )