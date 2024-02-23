import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from solve_model import solve_SIS, solve_SIR, solve_SEIR, solve_SEIRP, solve_SEIQR, solve_SEIQRDP

INFECTED_WEIGHT = 1

def SIS_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()

    coef_guess = [params['SIS']['beta']]
    bounds = [
        (0, 10),
    ]
    def objective_function(params, time_points, confirmed):
        I0 = confirmed[0]
        S0 = 1 - I0
        params = params.tolist()
        ini_values = [S0, I0]
        predicted_solution = solve_SIS(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 1]) ** 2) * INFECTED_WEIGHT / len(confirmed)

        loss = SSI
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed,), bounds=bounds)
    optimized_params = result.x

    params['SIS']['beta'] = optimized_params[0]

def SIR_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    coef_guess = [params['SIR']['beta'], params['SIR']['gamma']]
    bounds = [
        (0, 10),
        (0, 10),
    ]

    def objective_function(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - I0 - R0
        params = params.tolist()
        ini_values = [S0, I0, R0]
        predicted_solution = solve_SIR(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 1]) ** 2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 2]) ** 2) / len(recovered)

        loss = SSI + SSR
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered), bounds=bounds)
    optimized_params = result.x
    params['SIR']['beta'] = optimized_params[0]
    params['SIR']['gamma'] = optimized_params[1]

def SEI_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()

    coef_guess = [params['SEI']['beta'], params['SEI']['sigma']]
    bounds = [
        (0, 10),
        (0, 10),
    ]

    def objective_function(params, time_points, confirmed):
        I0 = confirmed[0]
        E0 = I0
        S0 = 1 - I0 - E0
        params = params.tolist()
        ini_values = [S0, E0, I0]
        predicted_solution = solve_SIR(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2) * INFECTED_WEIGHT / len(confirmed)

        loss = SSI
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed), bounds=bounds)
    optimized_params = result.x

    params['SEI']['beta'] = optimized_params[0]
    params['SEI']['sigma'] = optimized_params[1]


def SEIR_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    coef_guess = [params['SEIR']['beta'], params['SEIR']['sigma'], params['SEIR']['gamma']]  # beta, sigma, gamma
    bounds = [
        (0, 1),
        (0, 30),
        (0, 1)
    ]
    def objective_function(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - E0 - I0 - R0
        params = params.tolist()
        ini_values = [S0, E0, I0, R0]
        predicted_solution = solve_SEIR(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 3]) ** 2) / len(recovered)

        loss = SSI + SSR
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered), bounds=bounds)
    optimized_params = result.x

    params['SEIR']['beta'] = optimized_params[0]
    params['SEIR']['sigma'] = optimized_params[1]
    params['SEIR']['gamma'] = optimized_params[2]


def SEIRP_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    # alpha, beta, sigma, gamma
    coef_guess = [params['SEIRP']['alpha'], params['SEIRP']['beta'], params['SEIRP']['sigma'], params['SEIRP']['gamma']]
    bounds = [
        (0, 1),
        (0, 1),
        (0, 30),
        (0, 1)
    ]

    def objective_function(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - E0 - I0 - R0
        params = params.tolist()
        ini_values = [S0, E0, I0, R0, 0]
        predicted_solution = solve_SEIRP(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 3]) ** 2) / len(recovered)

        loss = SSI + SSR
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered), bounds=bounds)
    optimized_params = result.x

    params['SEIRP']['alpha'] = optimized_params[0]
    params['SEIRP']['beta'] = optimized_params[1]
    params['SEIRP']['sigma'] = optimized_params[2]
    params['SEIRP']['gamma'] = optimized_params[3]


def SEIQR_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()
    quarantined = data['Home_Confined'].to_numpy()

    # beta, sigma, gamma, lambda
    coef_guess = [params['SEIQR']['beta'], params['SEIQR']['sigma'], params['SEIQR']['gamma'],
                  params['SEIQR']['lambda']]
    bounds = [
        (0, 5),
        (0, 30),
        (0, 5),
        (0, 5)
    ]

    def objective_function(params, time_points, confirmed, quarantined, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        Q0 = quarantined[0]
        S0 = 1 - E0 - I0 - R0 - Q0
        params = params.tolist()
        ini_values = [S0, E0, I0, Q0, R0]
        predicted_solution = solve_SEIQR(time_points, ini_values + params)
        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 4]) ** 2) / len(recovered)
        SSQ = np.sum((quarantined - predicted_solution[:, 3]) ** 2) / len(quarantined)

        loss = SSI + SSR + SSQ
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, quarantined, recovered), bounds=bounds)
    optimized_params = result.x

    params['SEIQR']['beta'] = optimized_params[0]
    params['SEIQR']['sigma'] = optimized_params[1]
    params['SEIQR']['gamma'] = optimized_params[2]
    params['SEIQR']['lambda'] = optimized_params[3]


def SEIQRDP_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()
    quarantined = data['Home_Confined'].to_numpy()
    dead = data['Dead'].to_numpy()

    coef_guess = [params['SEIQRDP']['alpha'], params['SEIQRDP']['beta'], params['SEIQRDP']['sigma'],
                  params['SEIQRDP']['gamma'], params['SEIQRDP']['l1'], params['SEIQRDP']['l2'], params['SEIQRDP']['l3'],
                  params['SEIQRDP']['k1'], params['SEIQRDP']['k2'], params['SEIQRDP']['k3']]

    bounds = [
        (0, 1),
        (0, 5),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 100),
        (0, 1),
        (0, 1),
        (0, 100)
    ]

    def objective_function(params, time_points, confirmed, quarantined, recovered, dead):
        Q0 = quarantined[0]
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        D0 = dead[0]
        S0 = 1 - Q0 - E0 - R0 - D0 - I0
        params = params.tolist()
        ini_values = [S0, E0, I0, Q0, R0, D0, 0]
        predicted_solution = solve_SEIQRDP(time_points, ini_values + params)

        SSQ = np.sum((quarantined - predicted_solution[:, 3]) ** 2)
        SSR = np.sum((recovered - predicted_solution[:, 4]) ** 2)
        SSD = np.sum((dead - predicted_solution[:, 5]) ** 2)
        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2)
        loss = SSQ + SSR + SSD + SSI

        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, quarantined, recovered, dead),
                      bounds=bounds)
    optimized_params = result.x

    params['SEIQRDP']['alpha'] = optimized_params[0]
    params['SEIQRDP']['beta'] = optimized_params[1]
    params['SEIQRDP']['sigma'] = optimized_params[2]
    params['SEIQRDP']['gamma'] = optimized_params[3]
    params['SEIQRDP']['l1'] = optimized_params[4]
    params['SEIQRDP']['l2'] = optimized_params[5]
    params['SEIQRDP']['l3'] = optimized_params[6]
    params['SEIQRDP']['k1'] = optimized_params[7]
    params['SEIQRDP']['k2'] = optimized_params[8]
    params['SEIQRDP']['k3'] = optimized_params[9]
