from ast import Param
import numpy as np
import pvlib

from darkgreybox.base_model import DarkGreyModel, DarkGreyModelResult


class TiTeThRia(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTeThRia RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTeThRia.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Ch': {'value': 2.55, 'vary': False},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
        'Ria': {'value': 2},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    model = TiTeThRia(params, rec_duration=1).fit(X, y, method='nelder')
    ~~~~
    '''

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Te0' : Thermal envelope temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)
            - 'Rih' : Thermal resistance between internal and heating system
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
            - 'Ce' : Thermal capacitance of thermal envelope
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        Te : np.array
            Fitted thermal envelope temperature values
        Th : np.array
            Fitted heating system temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)
        Th = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']
        Th[0] = params['Th0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value
        Rih = params['Rih'].value
        Ria = params['Ria'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci) +
                   (Ta[i-1] - Ti[i-1]) / (Ria * Ci)) * self.rec_duration
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Te[i] = Te[i-1] + dTe
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti, 'Te': Te, 'Th': Th})


class TiTeTh(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTeTh RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTeTh.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Ch': {'value': 2.55, 'vary': False},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    model = TiTeTh(params, rec_duration=1).fit(X, y, method='nelder')
    ~~~~
    '''

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Te0' : Thermal envelope temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)
            - 'Rih' : Thermal resistance between internal and heating system
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
            - 'Ce' : Thermal capacitance of thermal envelope
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        Te : np.array
            Fitted thermal envelope temperature values
        Th : np.array
            Fitted heating system temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)
        Th = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']
        Th[0] = params['Th0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value
        Rih = params['Rih'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Te[i] = Te[i-1] + dTe
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti, 'Te': Te, 'Th': Th})


class TiTh(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTh RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTh.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
        'Th': df['Heating Circuit Temperature [˚C]'].values
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Th0': {'value': y[0], 'vary': False, 'min': 10, 'max': 80},
        'Ci': {'value': 132},
        'Ch': {'value': 2.55, 'vary': False},
        'Ria': {'value': 1},
        'Rih': {'value': 0.65, 'vary': False}
    }

    # fit using the Nelder-Mead method
    model = TiTh(params, rec_duration=1).fit(X, y, method='nelder')
    ~~~~
    '''

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Th0' : Heating system temperature at t(0)
            - 'Rih' : Thermal resistance between internal and heating system
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ch' : Thermal capacitance of heating system
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        Th : np.array
            Fitted heating system temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Th = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']

        Rih = params['Rih'].value
        Ria = params['Ria'].value

        Ci = params['Ci'].value
        Ch = params['Ch'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Ta[i-1] - Ti[i-1]) / (Ria * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci)) * self.rec_duration
            dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Th[i] = Th[i-1] + dTh

        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti, 'Th': Th})


class TiTe(DarkGreyModel):
    '''
    A DarkGrey Model representing a TiTe RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of TiTeTh.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Te0': {'value': y[0] - 2, 'vary': True, 'min': 10, 'max': 25},
        'Ci': {'value': 132},
        'Ce': {'value': 600},
        'Rie': {'value': 0.1},
        'Rea': {'value': 1},
    }

    # fit using the Nelder-Mead method
    model = TiTe(params, rec_duration=1).fit(X, y, method='nelder')
    ~~~~
    '''

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Te0' : Thermal envelope temperature at t(0)
            - 'Rie' : Thermal resistance between internal and thermal envelope
            - 'Rea' : Thermal resistance between thermal envelope and ambient
            - 'Ci' : Thermal capacitance of internal
            - 'Ce' : Thermal capacitance of thermal envelope
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        Te : np.array
            Fitted thermal envelope temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)
        Te = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0']
        Te[0] = params['Te0']

        Rie = params['Rie'].value
        Rea = params['Rea'].value

        Ci = params['Ci'].value
        Ce = params['Ce'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Ph[i-1]) / (Ci)) * self.rec_duration
            dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Te[i] = Te[i-1] + dTe

        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti, 'Te': Te})


class Ti(DarkGreyModel):
    '''
    A DarkGrey Model representing a Ti RC-equivalent circuit

    Notes
    -----
    See "Bacher & Madsen (2011) Identifying suitable models for the heat dynamics of buildings.
    Energy and Buildings. 43. 1511-1522. 10.1016/j.enbuild.2011.02.005." for a complete description
    of RC thermal models and the eqiuvalent circuit diagram of Ti.

    ~~~~
    # load data from e.g. pandas
    df = pd.read_csv()

    # assign internal temperature as the measured variable to be fitted to
    y = df['Internal Temperature [˚C]'].values

    # input values that will not change during the fit
    X = {
        'Ph': df['Boiler Power Output [kW]'].values,
        'Ta': df['Outside Air Temperature [˚C]'].values,
    }

    # parameters to be fitted
    # 'value' - initial value
    # 'min' & 'max' - boundaries
    # 'vary' - if false, the parameter will be fixed to its initial value
    params = {
        'Ti0': {'value': y[0], 'vary': False, 'min': 15, 'max': 25},
        'Ci': {'value': 132},
        'Ria': {'value': 1},
    }

    # fit using the Nelder-Mead method
    model = Ti(params, rec_duration=1).fit(X, y, method='nelder')
    ~~~~
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ic_param_names = ['Ti0']
        self.rc_param_names = ['Ci', 'Ria']
        self.input_param_names = ['Ta', 'Ph']

    def model(self, params, X):
        '''
        The system of differential equations describing the model

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Internal temperature at t(0)
            - 'Ria' : Thermal resistance between internal and ambient
            - 'Ci' : Thermal capacitance of internal
        X : dict
            - 'Ta' : List of ambient temperature values
            - 'Ph' : List of heating system power output values

        Returns
        -------
        Ti : np.array
            Fitted internal temperature values
        '''

        num_rec = len(X['Ta'])

        Ti = np.zeros(num_rec)

        # alias these params/X so that the differential equations look pretty
        Ti[0] = params['Ti0'].value

        Ria = params['Ria'].value
        Ci = params['Ci'].value

        Ta = X['Ta']
        Ph = X['Ph']

        for i in range(1, num_rec):

            # the model equations
            dTi = ((Ta[i-1] - Ti[i-1]) / (Ria * Ci) + (Ph[i-1]) / (Ci)) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi

        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti})
    

class TiTm2R1C(DarkGreyModel):
    '''
    A DarkGrey Model representing a 2R-1C Ti–Tm RC-equivalent circuit

    States
    ------
    Ti : Indoor air temperature
    Tm : Thermal mass (internal mass) temperature

    Inputs (X)
    ----------
    Ta      : Ambient/outdoor air temperature
    Q_heat  : Internal heat gains (people + equipment etc.)
    Q_vent  : Ventilation heat flow (can be negative when losing heat)
    Q_solar : Effective solar gains to the zone

    Parameters (params)
    -------------------
    Ti0 : Initial indoor air temperature
    Tm0 : Initial thermal mass temperature
    Ci  : Thermal capacitance of indoor air + light mass
    Cm  : Thermal capacitance of internal mass
    Rint: Thermal resistance between indoor air and internal mass
    Rout: Thermal resistance between indoor air and ambient
    '''

    def model(self, params, X):
        '''
        System of differential equations for the Ti–Tm 2R-1C model.

        Parameters
        ----------
        params : `lmfit.Parameters`
            - 'Ti0' : Indoor temperature at t(0)
            - 'Tm0' : Thermal mass temperature at t(0)
            - 'Ci'  : Indoor air capacitance
            - 'Cm'  : Thermal mass capacitance
            - 'Rint': Resistance between Ti and Tm
            - 'Rout': Resistance between Ti and Ta

        X : dict of np.array
            - 'Ta'     : Ambient temperature time series
            - 'Q_heat' : Internal heat gains time series
            - 'Q_vent' : Ventilation heat flow time series
            - 'Q_solar': Solar gains time series

        Returns
        -------
        DarkGreyModelResult
            Contains fitted Ti plus full state trajectories.
        '''

        num_rec = len(X['Ta'])

        # Allocate state arrays
        Ti = np.zeros(num_rec)
        Tm = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Tm[0] = params['Tm0']

        # Parameters
        Ci   = params['Ci'].value
        Cm   = params['Cm'].value
        Rint = params['Rint'].value
        Rout = params['Rout'].value

        # Inputs
        Ta     = X['Ta']
        Q_heat = X['Q_heat']
        Q_vent = X['Q_vent']
        Q_solar= X['Q_solar']

        # Time stepping (explicit Euler, as in your example)
        for i in range(1, num_rec):

            # dTi/dt = (Tm - Ti)/(Rint*Ci) + (Ta - Ti)/(Rout*Ci)
            #          + (Q_heat + Q_vent + Q_solar)/Ci
            dTi = (
                (Tm[i-1] - Ti[i-1]) / (Rint * Ci)
                + (Ta[i-1] - Ti[i-1]) / (Rout * Ci)
                + (Q_heat[i-1] + Q_vent[i-1] + Q_solar[i-1]) / Ci
            ) * self.rec_duration

            # dTm/dt = (Ti - Tm)/(Rint*Cm)
            dTm = (
                (Ti[i-1] - Tm[i-1]) / (Rint * Cm)
            ) * self.rec_duration

            Ti[i] = Ti[i-1] + dTi
            Tm[i] = Tm[i-1] + dTm

        # Return result with Ti as main output (like TiTeThRia)
        return DarkGreyModelResult(Ti, X, params, {'Ti': Ti, 'Tm': Tm})


class TiTmxvCN2R2C(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 2R2C thermal model (Ti, Th)
      - Water radiator driven by MVV and supply temperature
      - Ventilation, solar and internal gains
      - CO2-based grey-box occupancy sub-model

    States
    ------
    Ti  : Indoor air temperature (°C)
    Th  : Lumped thermal mass temperature (°C)
    x_v : Effective valve / flow opening (0–1)
    C   : Indoor CO2 concentration (ppm)
    N   : Effective occupancy (persons)

    Inputs X
    --------
    Ta        : Ambient temperature (°C)
    Tfor      : Supply water temperature to circuit (°C)
    Q_vent    : Ventilation heat flow (W)
    Q_solar   : Solar gains (W)
    MVV       : Heating command (%) for this radiator
    ACH       : Air changes per hour (1/h) or equivalent ventilation rate

    Parameters (params)
    -------------------
    Ti0, Th0, xv0, C0, N0 : Initial states
    Ci   : Air + light mass heat capacity (J/K)
    Ch   : Lumped mass heat capacity (J/K)
    Rint : Resistance between Ti and Th (K/W)
    Rout : Resistance between Ti and Ta (K/W)
    alpha: Radiator gain (W/K)
    tau_v: Valve time constant (s)
    q_pers      : Sensible heat per person (W/person)
    q_equip_var : Variable equipment gain per person (W/person)
    V           : Room volume (m3)
    E           : CO2 emission per person (ppm·m3/s/person or consistent units)
    C_out       : Outdoor CO2 concentration (ppm)
    Q_int_const : Constant internal gains (lights etc., W)
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Th  = np.zeros(num_rec)
        xv  = np.zeros(num_rec)
        C   = np.zeros(num_rec)
        N   = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']
        xv[0] = params['xv0']
        C[0]  = params['C0']
        N[0]  = params['N0']

        # Parameters
        Ci   = params['Ci'].value
        Ch   = params['Ch'].value
        Rint = params['Rint'].value
        Rout = params['Rout'].value
        alpha = params['alpha'].value
        tau_v = params['tau_v'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        V = params['V'].value
        E = params['E'].value
        C_out = params['C_out'].value
        Q_int_const = params['Q_int_const'].value

        # Inputs
        Ta        = X['Ta']
        Tfor      = X['Tfor']
        Q_vent    = X['Q_vent']
        Q_solar   = X['Q_solar']
        #Q_int_const = X['Q_int_const']  # already W (q_equip_const * A_room)
        MVV       = X['MVV']           # in %
        ACH       = X['ACH']           # 1/h
        #C_out     = X['C_out']

        dt = self.rec_duration  # [s]

        for k in range(1, num_rec):

            # 1) Valve / flow state: dxv/dt = (f(MVV) - xv)/tau_v
            f_MVV = MVV[k-1] / 100.0
            dxv = (f_MVV - xv[k-1]) / tau_v * dt
            xv[k] = xv[k-1] + dxv

            # 2) Radiator heat: Q_heat = alpha * xv * (Tfor - Th)
            Q_heat = alpha * xv[k-1] * (Tfor[k-1] - Th[k-1])

            # 3) Occupancy-based internal gains:
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int = Q_int_occ + Q_int_const

            # 4) Thermal states
            dTi = (
                (Th[k-1] - Ti[k-1]) / (Rint * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k-1] + Q_solar[k-1] + Q_int) / Ci
            ) * dt

            dTh = (
                (Ti[k-1] - Th[k-1]) / (Rint * Ch)
                + Q_heat / Ch
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Th[k] = Th[k-1] + dTh

            # 5) CO2–occupancy sub-model
            # Convert ACH [1/h] to s^-1
            ach_s = ACH[k-1] / 3600.0

            dC = (
                -ach_s * (C[k-1] - C_out)
                + (E / V) * N[k-1]
            ) * dt

            # Simple random-walk for N (no deterministic drift);
            # process noise handled in estimation, so deterministic part is zero.
            dN = 0.0

            C[k] = C[k-1] + dC
            N[k] = N[k-1] + dN

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Th': Th, 'xv': xv, 'C': C, 'N': N}
        )


class TiTmxvCN2R2C_KF(DarkGreyModel):
    """
    Grey-box model with Kalman Filter for occupancy estimation.
    
    Same as TiTmxvCN2R2C but with a simple Kalman filter that updates
    the occupancy N based on CO2 observations at each timestep.

    States
    ------
    Ti  : Indoor air temperature (°C)
    Th  : Lumped thermal mass temperature (°C)
    x_v : Effective valve / flow opening (0–1)
    C   : Indoor CO2 concentration (ppm) - predicted
    N   : Effective occupancy (persons) - updated via Kalman filter

    Additional Inputs X
    -------------------
    C_obs : Observed CO2 concentration for Kalman filter update (ppm)

    Additional Parameters
    ---------------------
    Q_N   : Process noise variance for N (persons²)
    R_C   : Measurement noise variance for CO2 (ppm²)
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Th  = np.zeros(num_rec)
        xv  = np.zeros(num_rec)
        C   = np.zeros(num_rec)
        N   = np.zeros(num_rec)
        P_N = np.zeros(num_rec)  # Variance of N estimate

        # Initial conditions
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']
        xv[0] = params['xv0']
        C[0]  = params['C0']
        N[0]  = params['N0']
        P_N[0] = params['P_N0'].value if 'P_N0' in params else 1.0  # Initial variance

        # Parameters
        Ci   = params['Ci'].value
        Ch   = params['Ch'].value
        Rint = params['Rint'].value
        Rout = params['Rout'].value
        alpha = params['alpha'].value
        tau_v = params['tau_v'].value
        q_pers = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        V = params['V'].value
        E = params['E'].value
        C_out = params['C_out'].value
        Q_int_const = params['Q_int_const'].value
        
        # Kalman filter noise parameters
        Q_N = params['Q_N'].value if 'Q_N' in params else 0.1   # Process noise for N
        R_C = params['R_C'].value if 'R_C' in params else 100.0  # Measurement noise for CO2

        # Inputs
        Ta        = X['Ta']
        Tfor      = X['Tfor']
        Q_vent    = X['Q_vent']
        Q_solar   = X['Q_solar']
        MVV       = X['MVV']
        ACH       = X['ACH']
        
        # CO2 observations for Kalman filter (use actual measured CO2)
        C_obs = X['C_obs'] if 'C_obs' in X else X.get('C0', C)

        dt = self.rec_duration

        for k in range(1, num_rec):
            # 1) Valve / flow state
            f_MVV = MVV[k-1] / 100.0
            dxv = (f_MVV - xv[k-1]) / tau_v * dt
            xv[k] = xv[k-1] + dxv

            # 2) Radiator heat
            Q_heat = alpha * xv[k-1] * (Tfor[k-1] - Th[k-1])

            # 3) Occupancy-based internal gains (use current N estimate)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int = Q_int_occ + Q_int_const

            # 4) Thermal states
            dTi = (
                (Th[k-1] - Ti[k-1]) / (Rint * Ci)
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)
                + (Q_vent[k-1] + Q_solar[k-1] + Q_int) / Ci
            ) * dt

            dTh = (
                (Ti[k-1] - Th[k-1]) / (Rint * Ch)
                + Q_heat / Ch
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Th[k] = Th[k-1] + dTh

            # 5) CO2–occupancy sub-model with Kalman Filter
            # We OBSERVE C (CO2), we want to ESTIMATE N (occupancy)
            
            # Get observed CO2 values
            C_obs_prev = C_obs[k-1] if hasattr(C_obs, '__getitem__') else C_obs
            C_obs_curr = C_obs[k] if hasattr(C_obs, '__getitem__') else C_obs
            
            # ACH is in 1/h, dt is in hours
            ach_h = ACH[k-1]  # Air changes per hour (1/h)
            
            # === KALMAN FILTER FOR N ===
            # The CO2 mass balance is: V * dC/dt = -Q*(C - C_out) + E_ppm*N
            # where:
            #   - C is in ppm (parts per million)
            #   - E is in m³ CO2/h/person (volume of pure CO2 emitted)
            #   - To convert E to ppm-equivalent: E_ppm = E * 10^6 ppm
            #   - Q = ACH * V (m³/h)
            #
            # Dividing by V: dC/dt = -ACH*(C - C_out) + (E*10^6/V)*N
            # Rearranging: N = (dC/dt + ACH*(C - C_out)) * V / (E * 10^6)
            
            # Step 1: PREDICT N (random walk model)
            N_pred = N[k-1]
            P_pred = P_N[k-1] + Q_N * dt  # Variance grows with process noise
            
            # Step 2: Compute "measured" N from observed CO2
            dC_obs = (C_obs_curr - C_obs_prev) / dt if dt > 0 else 0.0  # ppm/h
            ventilation_term = ach_h * (C_obs_prev - C_out)  # ppm/h
            
            # Convert E from m³/h to ppm-equivalent (E * 10^6)
            E_ppm = E * 1e6  # ppm·m³/h/person
            
            # N_meas = (ppm/h) * m³ / (ppm·m³/h/person) = persons
            N_meas = (dC_obs + ventilation_term) * V / E_ppm if E_ppm > 0 else 0.0
            
            # Step 3: MEASUREMENT UPDATE
            H = 1.0
            
            # Innovation (difference between measured and predicted N)
            y_innov = N_meas - N_pred
            
            # Innovation covariance: S = H*P_pred*H + R
            # R_C is measurement noise variance for N
            S = P_pred + R_C
            
            # Kalman gain
            K = P_pred / S if S > 1e-10 else 0.0
            
            # Update N estimate
            N[k] = N_pred + K * y_innov
            N[k] = max(0, N[k])  # N cannot be negative
            
            # Update variance
            P_N[k] = (1 - K) * P_pred
            
            # Store modeled CO2 (using observed values, since we observe C)
            C[k] = C_obs_curr

        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Th': Th, 'xv': xv, 'C': C, 'N': N, 'P_N': P_N}
        )


class TiThTmTrcn3R3C(DarkGreyModel):
    """
    Grey-box model of one room with:
      - 3R3C thermal model (Ti, Th, Tm)
      - Detailed water radiator (Tr driven by MVV and Tsup)
      - Ventilation via q_v (m3/h), solar and internal gains
      - CO2-based grey-box occupancy sub-model (c in fraction)

    States
    ------
    Ti  : Indoor air temperature (°C)
    Th  : Radiator/mean water temperature (°C)
    Tm  : Lumped thermal mass temperature (°C)
    Tr  : Return water temperature (°C)
    c   : Indoor CO2 concentration (volume fraction)
    N   : Effective occupancy (persons)
    Phi : Radiator flow

    Inputs X
    --------
    Ta      : Ambient temperature (°C)
    Tfor    : Supply water temperature to circuit (°C)
    Tsup    : Supply air temperature for ventilation (°C)
    qv      : Ventilation flow rate (m3/h)
    MVV     : Heating command (%) for this radiator
    Ik      : Irradiance
    c       : CO2 concentration (ppm)

    Parameters (params)
    ------------------
    Ti0, Th0, Tm0, Tr0, xv0, c0, N0 : Initial states
    Ci  : Air + light capacitance (J/K)
    Ch  : Radiator capacitance (J/K)
    Cm  : Thermal mass capacitance (J/K)
    Cf  : Radiator flow capacitance (J/K)
    Rim : Resistance Ti-Tm (K/W)
    Rout: Resistance Ti-Ta (K/W)
    Rih : Resistance Ti-Th (K/W)
    Rfr : Resistance Th-Tr (K/W)
    rho_air, cp_air: Air density (kg/m3), specific heat of air (J/kgK)
    Phi_max, cp_w : Max water flow (kg/s), water cp (J/kgK)
    V     : Room volume (m³)
    S     : Room surface (m²)
    A     : Window area (m²)
    G     : CO2 emission/person (m³/h/person)
    c_out : Outdoor CO2 fraction
    q_pers, q_equip_var : Gains/person (W/person)
    q_equip_const : Constant gains (W/m²)
    g : total solar energy transmittance of the glazing 
    """

    def model(self, params, X):
        num_rec = len(X['Ta'])

        # Allocate states
        Ti  = np.zeros(num_rec)
        Th  = np.zeros(num_rec)
        Tm  = np.zeros(num_rec)
        Tr  = np.zeros(num_rec)
        c   = np.zeros(num_rec)
        N   = np.zeros(num_rec)
        Phi = np.zeros(num_rec)

        # Initial conditions
        Ti[0] = params['Ti0']
        Th[0] = params['Th0']
        Tm[0] = params['Tm0']
        Tr[0] = params['Tr0']
        c[0]  = params['c0']
        N[0]  = params['N0']
        Phi[0]  = params['Phi0']

        # Parameters
        Ci   = params['Ci'].value
        Ch   = params['Ch'].value
        Cm   = params['Cm'].value
        Cf   = params['Cf'].value
        Rih = params['Rih'].value
        Rim = params['Rim'].value
        Rfr = params['Rfr'].value
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
        Phi_max = params['Phi_max'].value
        cp_w = params['cp_w'].value
        g = params['g'].value
    
        # Inputs
        Ta        = X['Ta']
        Tfor      = X['Tfor']
        Tsup      = X['Tsup']
        qv        = X['qv']
        MVV       = X['MVV']           # in %
        Ik        = X['Ik']
        c_meas    = X['c']             # Renamed to avoid overwriting state array
        
        dt = self.rec_duration  

        for k in range(1, num_rec):

            # 1) Normalized flow dynamics: dΦ/dt = (Φ_max * MVV - Φ)/C_f
            f_MVV = MVV[k-1] / 100.0
            dPhi = (Phi_max * f_MVV - Phi[k-1]) / Cf * dt
            Phi[k] = Phi[k-1] + dPhi

            # 2) Return water dynamics: dT_ret/dt = (T_h - T_ret)/(C_h * R_fr)
            dTret = ((Th[k-1] - Tr[k-1]) / Rfr) / Ch * dt
            Tr[k] = Tr[k-1] + dTret

            # 3) Current water mass flow & radiator heats
            m_w = Phi[k-1] * Phi_max  # kg/s
            Q_heat_in = m_w * cp_w * (Tfor[k-1] - Th[k-1])   # Supply→heater
            Q_heat = m_w * cp_w * (Th[k-1] - Tr[k-1])        # Heater→room

            # 4) Ventilation heat (q_v in m3/h → /3600 for m3/s)
            Q_vent = rho_air * cp_air * (qv[k-1]/3600) * (Tsup[k-1] - Ti[k-1])

            # 5) Internal gains (CO2 occupancy)
            Q_int_occ = (q_pers + q_equip_var) * N[k-1]
            Q_int = Q_int_occ + q_equip_const * S

            # 6) Solar gains
            Q_solar = g * A * Ik[k-1]

            # 7) Thermal states
            dTi = (
                (Th[k-1] - Ti[k-1]) / (Rih * Ci)      # Heater→air
                + (Tm[k-1] - Ti[k-1]) / (Rim * Ci)     # Mass→air  
                + (Ta[k-1] - Ti[k-1]) / (Rout * Ci)    # Air→ambient
                + (Q_vent + Q_solar + Q_int) / Ci      # Gains
            ) * dt

            dTh = (
                (Ti[k-1] - Th[k-1]) / (Rih * Ch)      # Air→heater (feedback)
                + Q_heat_in / Ch                       # Water→heater
            ) * dt

            dTm = (
                (Ti[k-1] - Tm[k-1]) / (Rim * Cm)      # Air↔mass
            ) * dt

            Ti[k] = Ti[k-1] + dTi
            Th[k] = Th[k-1] + dTh
            Tm[k] = Tm[k-1] + dTm

            # 8) CO2-occupancy (c in ppm)
            # dc/dt = G*N/V - (q_v/V)*(C - C_out) 
            # N from CO2 trajectory: N = V*qv/V*(C - C_out)/(E*10^6)
            # CO2 parameters
            dc = (
                1e6 * (G / V) * N[k-1]                          # Source: ppm/h
                - qv[k-1] / V * (c[k-1] - c_out)                # Sink: ppm/h
            ) * dt   

            c[k] = c[k-1] + dc

            # Occupancy from CO2 (deterministic, for reference/display)
            # N = V * qv / V * (C - C_out) / (E * 10^6) where E = G
            N_from_CO2 = V * qv[k-1] / V * (c[k-1] - c_out) / (G * 1e6)

            # Random walk for N state (noise handled in estimation)
            dN = 0.0  # or eta[k] if stochastic process
            N[k] = N[k-1] + dN


        return DarkGreyModelResult(
            Ti, X, params,
            {'Ti': Ti, 'Th': Th, 'Tm': Tm, 'Tr': Tr,  'c': c, 'N': N, 'Phi': Phi}
        )


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
        Q_inf = np.zeros(num_rec)


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
        q_inf = params['q_inf'].value  # Infiltration rate (air changes per hour)


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
            Q_inf[k] = q_inf * rho_air * cp_air * (V/3600) * (Ta[k-1] - Ti[k-1])


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
                + (Q_vent[k] + Q_solar[k] + Q_int[k] + Q_heat[k] + Q_inf[k]) / Ci  # All gains
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
