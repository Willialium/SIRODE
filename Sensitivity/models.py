import numpy as np
import pandas as pd

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP
from tuneModel import SEIR_fit, SEIRP_fit, SEIQR_fit, SEIQDRP_fit

pop = 6e7
data = pd.read_csv('../combined_data.csv')[
    ['Date', 'Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead', 'Region_Name']]
data = data.groupby('Date').sum().reset_index()
data[['Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead']] = data[['Total_Current_Positive_Cases',
                                                                                     'Recovered', 'Home_Confined',
                                                                                     'Dead']] / pop
data = data[data['Date'] >= '2020-03-01']
data = data[data['Date'] <= '2020-06-20']

scales = {
    'time': 1,
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
        'Xlambda': 1
    },
    'SEIQDRP': {
        'Xalpha': 1,
        'Xbeta': 1,
        'Xsigma': 1,
        'Xgamma': 1
    }
}
inis = {
    'E0': data['Total_Current_Positive_Cases'].iloc[0],
    'I0': data['Total_Current_Positive_Cases'].iloc[0],
    'R0': data['Recovered'].iloc[0],
    'P0': 0,
    'Q0': data['Home_Confined'].iloc[0],
    'D0': data['Dead'].iloc[0]
}

params = {
    'SEIR': {
        'beta': 0.14566402868860248,
        'sigma': 15.760004176832581,
        'gamma': 0.1190359988506418
    },
    'SEIRP': {
        'alpha': 0.07857877806639593,
        'beta': 0.7616387849526338,
        'sigma': 0.13889549106916346,
        'gamma': 0.024872143950524284
    },
    'SEIQR': {
        'beta': 0.7809947912506932,
        'sigma': 8.902459799240331,
        'gamma': 0.037156231730471424,
        'lambda': 0.7793507420805621
    },
    'SEIQDRP': {
        'alpha': .05116620636512935,
        'beta': .3611773188459532,
        'sigma': .708521695582455,
        'gamma': .03813449481479317,
        'l1': .04180630024314114,
        'l2': .2823859135168591,
        'l3': 13.088725432836508,
        'k1': .018195972150759264,
        'k2': 9.083826401618349e-10,
        'k3': 4.362924640962248
    }

}
SEIR_fit(data, params)
SEIRP_fit(data, params)
SEIQR_fit(data, params)
SEIQDRP_fit(data, params)

print(params)

time_interval = len(data['Date'])


def SEIR_test(use_SEIQDRP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0']

    ini = [S0, inis['E0'], inis['I0'], inis['R0']]
    fit_params = []
    if use_SEIQDRP:
        fit_params = [params['SEIQDRP']['beta'] * scales['SEIR']['Xbeta'],
                      params['SEIQDRP']['sigma'] * scales['SEIR']['Xsigma'],
                      params['SEIQDRP']['gamma'] * scales['SEIR']['Xgamma']]
    else:
        fit_params = [params['SEIR']['beta'] * scales['SEIR']['Xbeta'],
                      params['SEIR']['sigma'] * scales['SEIR']['Xsigma'],
                      params['SEIR']['gamma'] * scales['SEIR']['Xgamma']]

    ode_data = solve_ode_SEIR(time, ini + fit_params).T

    return ode_data


def SEIRP_test(use_SEIQDRP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['R0'], inis['P0']]

    fit_params = []

    if use_SEIQDRP:
        fit_params = [params['SEIQDRP']['alpha'] * scales['SEIRP']['Xalpha'],
                      params['SEIQDRP']['beta'] * scales['SEIRP']['Xbeta'],
                      params['SEIQDRP']['sigma'] * scales['SEIRP']['Xsigma'],
                      params['SEIQDRP']['gamma'] * scales['SEIRP']['Xgamma']]
    else:
        fit_params = [params['SEIRP']['alpha'] * scales['SEIRP']['Xalpha'],
                      params['SEIRP']['beta'] * scales['SEIRP']['Xbeta'],
                      params['SEIRP']['sigma'] * scales['SEIRP']['Xsigma'],
                      params['SEIRP']['gamma'] * scales['SEIRP']['Xgamma']]

    ode_data = solve_ode_SEIRP(time, ini + fit_params).T

    return ode_data


def SEIQR_test(use_SEIQDRP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['Q0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0']]

    fit_params = []
    if use_SEIQDRP:
        fit_params = [params['SEIQDRP']['beta'] * scales['SEIQR']['Xbeta'],
                      params['SEIQDRP']['sigma'] * scales['SEIQR']['Xsigma'],
                      params['SEIQDRP']['gamma'] * scales['SEIQR']['Xgamma'],
                      params['SEIQDRP']['lambda'] * scales['SEIQR']['Xlambda']]

    else:
        fit_params = [params['SEIQR']['beta'] * scales['SEIQR']['Xbeta'],
                      params['SEIQR']['sigma'] * scales['SEIQR']['Xsigma'],
                      params['SEIQR']['gamma'] * scales['SEIQR']['Xgamma'],
                      params['SEIQR']['lambda'] * scales['SEIQR']['Xlambda']]
    ode_data = solve_ode_SEIQR(time, ini + fit_params).T

    return ode_data


def SEIQDRP_test():
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['Q0'] - inis['E0'] - inis['R0'] - inis['D0'] - inis['I0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0'], inis['D0'], inis['P0']]

    fit_params = [params['SEIQDRP']['alpha'] * scales['SEIQDRP']['Xalpha'],
                  params['SEIQDRP']['beta'] * scales['SEIQDRP']['Xbeta'],
                  params['SEIQDRP']['sigma'] * scales['SEIQDRP']['Xsigma'],
                  params['SEIQDRP']['gamma'] * scales['SEIQDRP']['Xgamma'],
                  params['SEIQDRP']['l1'], params['SEIQDRP']['l2'], params['SEIQDRP']['l3'],
                  params['SEIQDRP']['k1'], params['SEIQDRP']['k2'], params['SEIQDRP']['k3']]

    ode_data = solve_ode_SEIQDRP(time, ini + fit_params).T

    return ode_data
