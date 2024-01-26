import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from solve_model import solve_ode

def workit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy()
    recovered = data['Recovered'].to_numpy()

    coef_guess = [.04, .03, .01] # beta, delta, gamma
    bounds = [
        (0,1),
        (0,5),
        (0,1)
    ]
    def objective_function(params, time_points, confirmed, recovered, N):
        I0 = confirmed[0]
        E0 = .3 * confirmed[0]
        R0 = recovered[0]
        S0 = N - E0 - I0 - R0
        params = params.tolist() + [N]
        ini_values = [S0, E0, I0, R0]
        predicted_solution = solve_ode(time_points, ini_values + params)

        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2) / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 3]) ** 2) / len(recovered)

        loss = SSI + SSR
        #print('loss:', loss, SSI, SSR)
        return loss
    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered, pop), bounds=bounds)
    optimized_params = result.x

    return optimized_params.tolist()
