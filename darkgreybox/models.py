import numpy as np

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
    Q_int_const : Constant internal gains (lights etc., W)
    MVV       : Heating command (%) for this radiator
    ACH       : Air changes per hour (1/h) or equivalent ventilation rate
    C_out     : Outdoor CO2 concentration (ppm)

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
        S = 20      # m²
        V = 2.5*S   # m³
        E = 0.016   # m³/h
        Ci   = params['Ci'].value
        Ch   = params['Ch'].value
        Rint = params['Rint'].value
        Rout = params['Rout'].value
        alpha = params['alpha'].value
        tau_v = params['tau_v'].value
        q_pers      = params['q_pers'].value
        q_equip_var = params['q_equip_var'].value
        V      = params['V'].value
        E           = params['E'].value
        C_out = params['C_out'].value

        # Inputs
        Ta        = X['Ta']
        Tfor      = X['Tfor']
        Q_vent    = X['Q_vent']
        Q_solar   = X['Q_solar']
        Q_int_const = X['Q_int_const']  # already W (q_equip_const * A_room)
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
            Q_int = Q_int_occ + Q_int_const[k-1]

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