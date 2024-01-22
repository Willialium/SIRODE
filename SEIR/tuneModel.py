import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from get_state_data import get_cleaned_data
from get_country_data import get_cleaned_country_data, getN
from solve_model import solve_ode

def workit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Confirmed'].to_numpy()
    recovered = data['Recovered'].to_numpy()
    dead = data['Dead'].to_numpy()

    coef_guess = [.04, .03, .01] # beta, delta, gamma
    bounds = [
        (0,1),
        (0,1),
        (0,1)
    ]
    def objective_function(params, time_points, confirmed, recovered, dead, N):
        quarantined = confirmed - recovered - dead
        Q0 = quarantined[0]
        I0 = .3 * Q0
        E0 = .3 * Q0
        R0 = recovered[0]
        D0 = dead[0]
        S0 = N - Q0 - E0 - R0 - D0 - I0
        params = params.tolist() + [N]
        ini_values = [S0, E0, I0, Q0, R0, D0, 0]
        predicted_solution = solve_ode(time_points, ini_values + params)

        SSQ = np.sum((quarantined - predicted_solution[:, 3]) ** 2)
        SSR = np.sum((recovered - predicted_solution[:, 4]) ** 2)
        SSD = np.sum((dead - predicted_solution[:, 5]) ** 2)

        loss = SSQ + SSR + SSD
        #print('loss:', loss, SSQ, SSR, SSD)
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered, dead, pop), bounds=bounds)
    optimized_params = result.x

    return optimized_params.tolist()
