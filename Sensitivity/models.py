import matplotlib.pyplot as plt
import numpy as np

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP

time = np.array([i for i in range(0, 140)])
scales = {
    'SEIR': {
        'Xbeta': 1,
        'Xsigma': 1,
        'Xgamma': 1,
    },
    'SEIRP': {
        'Xbeta': 1,
        'Xsigma': 1,
        'Xgamma': 1,
        'Xalpha': 1
    }
}
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


def SEIR_test(Xbeta = None, Xsigma = None, Xgamma = None):
    if Xbeta is not None:
        scales['SEIR']['Xbeta'] = Xbeta
    if Xsigma is not None:
        scales['SEIR']['Xsigma'] = Xsigma
    if Xgamma is not None:
        scales['SEIR']['Xgamma'] = Xgamma


    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0']

    ini = [S0, inis['E0'], inis['I0'], inis['R0']]

    fit_params = [params['beta']*scales['SEIR']['Xbeta'], params['sigma']*scales['SEIR']['Xsigma'], params['gamma']*scales['SEIR']['Xgamma']]

    ode_data = solve_ode_SEIR(time, ini + fit_params).T
    print(scales['SEIR'])
    return ode_data


def SEIRP_test(Xbeta = None, Xsigma = None, Xgamma = None):

    if Xbeta is not None:
        scales['SEIRP']['Xbeta'] = Xbeta
    if Xsigma is not None:
        scales['SEIRP']['Xsigma'] = Xsigma
    if Xgamma is not None:
        scales['SEIRP']['Xgamma'] = Xgamma

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['R0'], inis['P0']]

    fit_params = [params['alpha']*scales['SEIRP']['Xalpha'], params['beta']*scales['SEIRP']['Xbeta'], params['sigma']*scales['SEIRP']['Xsigma'], params['gamma']*scales['SEIRP']['Xgamma']]

    ode_data = solve_ode_SEIRP(time, ini + fit_params).T

    return ode_data


def SEIQR_test():
    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['Q0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0']]

    fit_params = [params['beta'], params['sigma'], params['gamma'], params['laambda']]
    ode_data = solve_ode_SEIQR(time, ini + fit_params).T

    return ode_data

def SEIQDRP_test():
    S0 = 1 - inis['Q0'] - inis['E0'] - inis['R0'] - inis['D0'] - inis['I0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0'], inis['P0'], inis['D0']]

    fit_params = [params['alpha'], params['beta'], params['sigma'], params['gamma'], params['l1'], params['l2'],
                  params['l3'], params['k1'], params['k2'], params['k3']]

    ode_data = solve_ode_SEIQDRP(time, ini + fit_params).T

    return ode_data
