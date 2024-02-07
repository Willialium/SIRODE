import matplotlib.pyplot as plt
import numpy as np

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP

time = np.linspace(0, 140)

inis = {
    'E0': 5e-5,
    'I0': 5e-5,
    'R0': 2.77e-6,
    'P0': 0,
    'Q0': 2.6e-5,
    'D0': 1.13e-6
}
params = {
    'alpha': .079,
    'beta': .76,
    'sigma': .14,
    'gamma': .025,
    'laambda': .661,
    'l1': .042,
    'l2': .28,
    'l3': 13,
    'k1': .018,
    'k2': 7.5e-10,
    'k3': 4.36
}


def SEIR_test(Xbeta = 1, subplot=1):
    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0']

    ini = [S0, inis['E0'], inis['I0'], inis['R0']]

    fit_params = [params['beta']*Xbeta, params['sigma'], params['gamma']]

    ode_data = solve_ode_SEIR(time, ini + fit_params)

    return ode_data


def SEIRP_test(subplot=2):
    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['R0'], inis['P0']]

    fit_params = [params['alpha'], params['beta'], params['sigma'], params['gamma']]

    ode_data = solve_ode_SEIRP(time, ini + fit_params)

    return ode_data


def SEIQR_test(subplot=3):
    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['Q0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0']]

    fit_params = [params['beta'], params['sigma'], params['gamma'], params['laambda']]
    ode_data = solve_ode_SEIQR(time, ini + fit_params)

    return ode_data

def SEIQDRP_test(subplot=4):
    S0 = 1 - inis['Q0'] - inis['E0'] - inis['R0'] - inis['D0'] - inis['I0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0'], inis['P0'], inis['D0']]

    fit_params = [params['alpha'], params['beta'], params['sigma'], params['gamma'], params['l1'], params['l2'],
                  params['l3'], params['k1'], params['k2'], params['k3']]

    ode_data = solve_ode_SEIQDRP(time, ini + fit_params)

    return ode_data
