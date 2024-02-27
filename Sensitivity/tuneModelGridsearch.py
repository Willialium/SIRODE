import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product

from solve_model import solve_SIS, solve_SIR, solve_SEIR, solve_SEIRP, solve_SEIQR, solve_SEIQRDP

INFECTED_WEIGHT = 1

def SIS_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()

    max_I_index = np.argmax(confirmed)
    max_I = confirmed[max_I_index]

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

    max_I_index = np.argmax(confirmed)
    max_I = confirmed[max_I_index]

    # Define the grid for parameter search
    beta_values = np.linspace(0, 2, 41)  # Adjust the number of values and range as needed
    gamma_values = np.linspace(0, 2, 41)  # Adjust the number of values and range as needed

    best_loss = float('inf')
    best_params = {}

    for beta, gamma in product(beta_values, gamma_values):
        def objective_function(beta, gamma, time_points):
            I0 = confirmed[0]
            R0 = recovered[0]
            S0 = 1 - I0 - R0
            params = [beta, gamma]
            ini_values = [S0, I0, R0]
            predicted_solution = solve_SIR(time_points, ini_values + params)


            loss = np.abs((np.max(predicted_solution[:,1])-max_I)/max_I)*40 + np.abs(np.argmax(predicted_solution[:,1]) - max_I_index)
            return loss

        loss = objective_function(beta, gamma, time)

        # Update best parameters if current parameters yield lower loss
        if loss < best_loss:
            best_loss = loss
            best_params['beta'] = beta
            best_params['gamma'] = gamma

    params['SIR']['beta'] = best_params['beta']
    params['SIR']['gamma'] = best_params['gamma']

def SEI_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()

    max_I_index = np.argmax(confirmed)
    max_I = confirmed[max_I_index]

    beta_values = np.linspace(0, 2, 41)  # Adjust the number of values and range as needed
    sigma_values = np.linspace(0, 2, 41)  # Adjust the number of values and range as needed

    best_loss = float('inf')
    best_params = {}

    for beta, sigma in product(beta_values, sigma_values):
        def objective_function(beta, sigma, time_points):
            I0 = confirmed[0]
            E0 = I0
            S0 = 1 - I0 - E0
            params = [beta,sigma]
            ini_values = [S0, E0, I0]
            predicted_solution = solve_SIR(time_points, ini_values + params)

            loss = np.abs((np.max(predicted_solution[:, 2]) - max_I) / max_I) * 40 + np.abs(
                np.argmax(predicted_solution[:, 2]) - max_I_index)

            return loss

        loss = objective_function(beta, sigma, time)

        if loss < best_loss:
            best_loss = loss
            best_params['beta'] = beta
            best_params['sigma'] = sigma
    params['SEI']['beta'] = best_params['beta']
    params['SEI']['sigma'] = best_params['sigma']


def SEIR_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    max_I_index = np.argmax(confirmed)
    max_I = confirmed[max_I_index]

    beta_values = np.linspace(2, 4, 41)  # Adjust the number of values and range as needed
    sigma_values = np.linspace(0, 1, 11)  # Adjust the number of values and range as needed
    gamma_values = np.linspace(2, 4, 41)  # Adjust the number of values and range as needed

    best_loss = float('inf')
    best_params = {}

    for beta, sigma, gamma in product(beta_values, sigma_values, gamma_values):
        def objective_function(beta, sigma, gamma, time_points):
            I0 = confirmed[0]
            E0 = I0
            R0 = recovered[0]
            S0 = 1 - I0 - E0 - R0
            params = [beta,sigma, gamma]
            ini_values = [S0, E0, I0, R0]
            predicted_solution = solve_SEIR(time_points, ini_values + params)

            loss = np.abs((np.max(predicted_solution[:, 2]) - max_I) / max_I) * 40 + np.abs(
                np.argmax(predicted_solution[:, 2]) - max_I_index)

            return loss

        loss = objective_function(beta, sigma, gamma, time)

        if loss < best_loss:
            best_loss = loss
            best_params['beta'] = beta
            best_params['sigma'] = sigma
            best_params['gamma'] = gamma

    params['SEIR']['beta'] = best_params['beta']
    params['SEIR']['sigma'] = best_params['sigma']
    params['SEIR']['gamma'] = best_params['gamma']


def SEIRP_fit(data, params):
    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    max_I_index = np.argmax(confirmed)
    max_I = confirmed[max_I_index]

    alpha_values = np.linspace(0, 1, 13)  # Adjust the number of values and range as needed
    beta_values = np.linspace(0, 1, 41)  # Adjust the number of values and range as needed
    sigma_values = np.linspace(0, 1, 11)  # Adjust the number of values and range as needed
    gamma_values = np.linspace(0, 1, 41)  # Adjust the number of values and range as needed

    best_loss = float('inf')
    best_params = {}
    for alpha, beta, sigma, gamma in product(alpha_values, beta_values, sigma_values, gamma_values):
        def objective_function(alpha, beta, sigma, gamma, time_points):
            I0 = confirmed[0]
            E0 = I0
            R0 = recovered[0]
            S0 = 1 - I0 - E0 - R0
            params = [alpha, beta, sigma, gamma]
            ini_values = [S0, E0, I0, R0, 0]
            predicted_solution = solve_SEIRP(time_points, ini_values + params)

            I_loss = np.abs((np.max(predicted_solution[:, 2]) - max_I) / max_I) * 40 + np.abs(
                np.argmax(predicted_solution[:, 2]) - max_I_index)

            R_loss = np.abs((predicted_solution[:, 3][-1] - recovered[-1]) / recovered[-1])

            return I_loss + R_loss

        loss = objective_function(alpha, beta, sigma, gamma, time)

        if loss < best_loss:
            best_loss = loss
            best_params['alpha'] = alpha
            best_params['beta'] = beta
            best_params['sigma'] = sigma
            best_params['gamma'] = gamma

    params['SEIRP']['alpha'] = best_params['alpha']
    params['SEIRP']['beta'] = best_params['beta']
    params['SEIRP']['sigma'] = best_params['sigma']
    params['SEIRP']['gamma'] = best_params['gamma']


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
