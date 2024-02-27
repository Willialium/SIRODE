import numpy as np
import pandas as pd

from solve_model import solve_SIS, solve_SIR, solve_SEI, solve_SEIR, solve_SEIRP, solve_SEIQR, solve_SEIQRDP
from tuneModel import SIS_fit, SEIQR_fit, SEIQRDP_fit
from tuneModelGridsearch import SIR_fit, SEI_fit, SEIR_fit, SEIRP_fit

pop = 6e7
data = pd.read_csv('../combined_data.csv')[
    ['Date', 'Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead', 'Region_Name']]
data = data.groupby('Date').sum().reset_index()
data[['Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead']] = data[['Total_Current_Positive_Cases',
                                                                                     'Recovered', 'Home_Confined',
                                                                                     'Dead']] / pop

# First wave is 2020-03-01 to 2020-06-30
# Second wave is 2020-10-01 to 2021-07-01
### There is a spike at ~ 2021-02-30

data = data[data['Date'] >= '2020-03-01']
data = data[data['Date'] <= '2020-07-30']

scales = {
    'time': 1,
    'SIS': {
        'Xbeta': 1
    },
    'SIR': {
        'Xbeta': 1,
        'Xgamma': 1
    },
    'SEI': {
        'Xbeta': 1,
        'Xsigma': 1
    },
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
    'SEIQRDP': {
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
    'SIS': {
        'beta': 0.14566402868860248
    },
    'SIR': {
        'beta': 1.25,
        'gamma': 1.14
    },
    'SEI': {
        'beta': 0.14566402868860248,
        'sigma': 15.760004176832581
    },
    'SEIR': {
        'beta': 3.3,
        'sigma': .73,
        'gamma': 2.6
    },
    'SEIRP': {
        'alpha': 0.08,
        'beta': 0.98,
        'sigma': 0.1,
        'gamma': 0.03
    },
    'SEIQR': {
        'beta': 0.7809947912506932,
        'sigma': 8.902459799240331,
        'gamma': 0.037156231730471424,
        'lambda': 0.7793507420805621
    },
    'SEIQRDP': {
        'alpha': .05116620636512935,
        'beta': .3611773188459532,
        'sigma': .708521695582455,
        'gamma': .03813449481479317,
        'lambda': 0.7652522426471923,
        'l1': .04180630024314114,
        'l2': .2823859135168591,
        'l3': 13.088725432836508,
        'k1': .018195972150759264,
        'k2': 9.083826401618349e-10,
        'k3': 4.362924640962248
    }

}

SIS_fit(data, params)
SIR_fit(data, params)
SEI_fit(data, params)
SEIR_fit(data, params)
SEIRP_fit(data, params)
SEIQR_fit(data, params)
SEIQRDP_fit(data, params)

time_interval = len(data['Date'])

def SIS_test(use_SEIQRDP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['I0']

    ini = [S0, inis['I0']]
    fit_params = []
    if use_SEIQRDP:
        fit_params = [params['SEIQDRP']['beta'] * scales['SIS']['Xbeta']]
    else:
        fit_params = [params['SIS']['beta'] * scales['SIS']['Xbeta']]
    ode_data = solve_SIS(time, ini + fit_params).T

    return ode_data

def SIR_test(use_SEIQRDP=False):
    time = np.arange(time_interval * scales['time'])
    S0 = 1 - inis['I0'] - inis['R0']
    ini = [S0, inis['I0'], inis['R0']]
    if use_SEIQRDP:
        fit_params = [params['SEIQDRP']['beta'] * scales['SIR']['Xbeta'],
                      params['SEIQDRP']['gamma'] * scales['SIR']['Xgamma']]
    else:
        fit_params = [params['SIR']['beta'] * scales['SIR']['Xbeta'], params['SIR']['gamma'] * scales['SIR']['Xgamma']]
    ode_data = solve_SIR(time, ini + fit_params).T
    print(params['SIR'])
    return ode_data

def SEI_test(use_SEIQRDP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['I0'] - inis['E0']
    ini = [S0, inis['E0'], inis['I0']]

    fit_params = []
    if use_SEIQRDP:
        fit_params = [params['SEIQDRP']['beta'] * scales['SIR']['Xbeta'],
                      params['SEIQDRP']['gamma'] * scales['SIR']['Xgamma']]
    else:
        fit_params = [params['SEI']['beta'] * scales['SEI']['Xbeta'], params['SEI']['sigma'] * scales['SEI']['Xsigma']]
    ode_data = solve_SEI(time, ini + fit_params).T

    return ode_data

def SEIR_test(use_SEIQRDP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0']

    ini = [S0, inis['E0'], inis['I0'], inis['R0']]
    fit_params = []
    if use_SEIQRDP:
        fit_params = [params['SEIQRDP']['beta'] * scales['SEIR']['Xbeta'],
                      params['SEIQRDP']['sigma'] * scales['SEIR']['Xsigma'],
                      params['SEIQRDP']['gamma'] * scales['SEIR']['Xgamma']]
    else:
        fit_params = [params['SEIR']['beta'] * scales['SEIR']['Xbeta'],
                      params['SEIR']['sigma'] * scales['SEIR']['Xsigma'],
                      params['SEIR']['gamma'] * scales['SEIR']['Xgamma']]
    ode_data = solve_SEIR(time, ini + fit_params).T
    return ode_data


def SEIRP_test(use_SEIQRDP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['R0'], inis['P0']]

    fit_params = []

    if use_SEIQRDP:
        fit_params = [params['SEIQRDP']['alpha'] * scales['SEIRP']['Xalpha'],
                      params['SEIQRDP']['beta'] * scales['SEIRP']['Xbeta'],
                      params['SEIQRDP']['sigma'] * scales['SEIRP']['Xsigma'],
                      params['SEIQRDP']['gamma'] * scales['SEIRP']['Xgamma']]
    else:
        fit_params = [params['SEIRP']['alpha'] * scales['SEIRP']['Xalpha'],
                      params['SEIRP']['beta'] * scales['SEIRP']['Xbeta'],
                      params['SEIRP']['sigma'] * scales['SEIRP']['Xsigma'],
                      params['SEIRP']['gamma'] * scales['SEIRP']['Xgamma']]

    ode_data = solve_SEIRP(time, ini + fit_params).T

    return ode_data


def SEIQR_test(use_SEIQRDP=False):
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['E0'] - inis['I0'] - inis['R0'] - inis['Q0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0']]

    fit_params = []
    if use_SEIQRDP:
        fit_params = [params['SEIQRDP']['beta'] * scales['SEIQR']['Xbeta'],
                      params['SEIQRDP']['sigma'] * scales['SEIQR']['Xsigma'],
                      params['SEIQRDP']['gamma'] * scales['SEIQR']['Xgamma'],
                      params['SEIQRDP']['lambda'] * scales['SEIQR']['Xlambda']]

    else:
        fit_params = [params['SEIQR']['beta'] * scales['SEIQR']['Xbeta'],
                      params['SEIQR']['sigma'] * scales['SEIQR']['Xsigma'],
                      params['SEIQR']['gamma'] * scales['SEIQR']['Xgamma'],
                      params['SEIQR']['lambda'] * scales['SEIQR']['Xlambda']]
    ode_data = solve_SEIQR(time, ini + fit_params).T

    return ode_data

def SEIQRDP_test():
    time = np.arange(0, time_interval * scales['time'])

    S0 = 1 - inis['Q0'] - inis['E0'] - inis['R0'] - inis['D0'] - inis['I0'] - inis['P0']
    ini = [S0, inis['E0'], inis['I0'], inis['Q0'], inis['R0'], inis['D0'], inis['P0']]

    fit_params = [params['SEIQRDP']['alpha'] * scales['SEIQRDP']['Xalpha'],
                  params['SEIQRDP']['beta'] * scales['SEIQRDP']['Xbeta'],
                  params['SEIQRDP']['sigma'] * scales['SEIQRDP']['Xsigma'],
                  params['SEIQRDP']['gamma'] * scales['SEIQRDP']['Xgamma'],
                  params['SEIQRDP']['l1'], params['SEIQRDP']['l2'], params['SEIQRDP']['l3'],
                  params['SEIQRDP']['k1'], params['SEIQRDP']['k2'], params['SEIQRDP']['k3']]

    ode_data = solve_SEIQRDP(time, ini + fit_params).T

    return ode_data
