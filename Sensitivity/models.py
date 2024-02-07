import matplotlib.pyplot as plt
import numpy as np

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP

time = np.array([i for i in range(0, 142)])
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
    },
    'SEIQR': {
        'Xbeta': 1,
        'Xsigma': 1,
        'Xgamma': 1,
        'Xlaambda': 1
    },
    'SEIQDRP': {
        'Xalpha': 1,
        'Xbeta': 1,
        'Xsigma': 1,
        'Xgamma': 1
    }
}
inis = {
    'E0': 5.2566666666666666e-05,
    'I0':  5.2566666666666666e-05,
    'R0': 2.7666666666666667e-06,
    'P0': 0,
    'Q0': 2.66e-05,
    'D0': 1.1333333333333334e-06
}
params = {
    'alpha': .05116620636512935,
    'beta': .3611773188459532,
    'sigma': .708521695582455,
    'gamma': .03813449481479317,
    'laambda': .661,
    'l1': .04180630024314114,
    'l2': .2823859135168591,
    'l3': 13.088725432836508,
    'k1': .018195972150759264,
    'k2': 9.083826401618349e-10,
    'k3': 4.362924640962248
}


def SEIR_test():

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0']

    ini = [S0, inis['E0'], inis['I0'], inis['R0']]

    fit_params = [params['beta']*scales['SEIR']['Xbeta'], params['sigma']*scales['SEIR']['Xsigma'], params['gamma']*scales['SEIR']['Xgamma']]

    ode_data = solve_ode_SEIR(time, ini + fit_params).T
    return ode_data


def SEIRP_test():

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['R0'], inis['P0']]

    fit_params = [params['alpha']*scales['SEIRP']['Xalpha'], params['beta']*scales['SEIRP']['Xbeta'], params['sigma']*scales['SEIRP']['Xsigma'], params['gamma']*scales['SEIRP']['Xgamma']]

    ode_data = solve_ode_SEIRP(time, ini + fit_params).T

    return ode_data


def SEIQR_test():
    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['Q0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0']]

    fit_params = [params['beta']*scales['SEIQR']['Xbeta'], params['sigma']*scales['SEIQR']['Xsigma'], params['gamma']*scales['SEIQR']['Xgamma'], params['laambda']*scales['SEIQR']['Xlaambda']]
    ode_data = solve_ode_SEIQR(time, ini + fit_params).T

    return ode_data

def SEIQDRP_test():
    S0 = 1 - inis['Q0'] - inis['E0'] - inis['R0'] - inis['D0'] - inis['I0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0'], inis['D0'], inis['P0']]

    fit_params = [params['alpha']*scales['SEIQDRP']['Xalpha'], params['beta']*scales['SEIQDRP']['Xbeta'], params['sigma']*scales['SEIQDRP']['Xsigma'], params['gamma']*scales['SEIQDRP']['Xgamma'], params['l1'], params['l2'],
                  params['l3'], params['k1'], params['k2'], params['k3']]

    ode_data = solve_ode_SEIQDRP(time, ini + fit_params).T

    return ode_data
