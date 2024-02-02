import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from solve_model import solve_ode_SEIR,solve_ode_SEIRP, solve_ode_SEIQR,solve_ode_SEIQDRP

INFECTED_WEIGHT = 1

def SEIR_fit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy() / pop
    recovered = data['Recovered'].to_numpy() / pop

    coef_guess = [0.04, 15, 0.01] # beta, sigma, gamma
    bounds = [
        (0,5),
        (0,50),
        (0,5)
    ]
    def objective_function(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - E0 - I0 - R0
        params = params.tolist()
        ini_values = [S0, E0, I0, R0]
        predicted_solution = solve_ode_SEIR(time_points, ini_values + params)

        #SSI = np.sum(np.abs(confirmed - predicted_solution[:, 2])) * INFECTED_WEIGHT / len(confirmed)
        #SSR = np.sum(np.abs(recovered - predicted_solution[:, 3])) / len(recovered)

        SSI = np.sum((confirmed - predicted_solution[:, 2])**2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 3])**2) / len(recovered)

        loss = SSI + SSR
        return loss

    def objective_function2(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - E0 - I0 - R0
        params = params.tolist()
        ini_values = [S0, E0, I0, R0]
        predicted_solution = solve_ode_SEIR(time_points, ini_values + params)

        # calculate Pearson correlation for confirmed cases
        corr_infected = np.corrcoef(confirmed, predicted_solution[:, 2])[0, 1]

        # calculate Pearson correlation for recovered cases
        corr_recovered = np.corrcoef(recovered, predicted_solution[:, 3])[0, 1]

        # use negative correlation as loss
        loss = - (corr_infected + corr_recovered) / 2

        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered), bounds=bounds)
    # result = differential_evolution(objective_function, bounds, args=(time, confirmed, recovered, pop))
    optimized_params = result.x

    return optimized_params.tolist()
def SEIRP_fit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy() / pop
    recovered = data['Recovered'].to_numpy() / pop

    coef_guess = [0.07861792711039865, 0.7616354118609561, 0.13888766257731705, 0.024868741025170418] # beta, sigma, gamma
    bounds = [
        (0, 5),
        (0,5),
        (0,100),
        (0,5)
    ]
    def objective_function(params, time_points, confirmed, recovered):
        I0 = confirmed[0]
        E0 = confirmed[0]
        R0 = recovered[0]
        S0 = 1 - E0 - I0 - R0
        params = params.tolist()
        ini_values = [S0, E0, I0, R0, 0]
        predicted_solution = solve_ode_SEIRP(time_points, ini_values + params)

        #SSI = np.sum(np.abs(confirmed - predicted_solution[:, 2])) * INFECTED_WEIGHT / len(confirmed)
        #SSR = np.sum(np.abs(recovered - predicted_solution[:, 3])) / len(recovered)

        SSI = np.sum((confirmed - predicted_solution[:, 2])**2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 3])**2) / len(recovered)

        loss = SSI + SSR
        return loss
    result = minimize(objective_function, coef_guess, args=(time, confirmed, recovered), bounds=bounds)
    # result = differential_evolution(objective_function, bounds, args=(time, confirmed, recovered, pop))
    optimized_params = result.x

    return optimized_params.tolist()
def SEIQR_fit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy() / pop
    recovered = data['Recovered'].to_numpy() / pop
    quarantined = data['Home_Confined'].to_numpy() / pop

    coef_guess = [0.7616354118609561, 0.13888766257731705, 0.024868741025170418, .12766866818704564] # beta, sigma, gamma, delta
    bounds = [
        (0,50),
        (0,100),
        (0,5),
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
        predicted_solution = solve_ode_SEIQR(time_points, ini_values + params)

        #SSI = np.sum(np.abs(confirmed - predicted_solution[:, 2])) * INFECTED_WEIGHT / len(confirmed)
        #SSR = np.sum(np.abs(recovered - predicted_solution[:, 3])) / len(recovered)

        SSI = np.sum((confirmed - predicted_solution[:, 2])**2) * INFECTED_WEIGHT / len(confirmed)
        SSR = np.sum((recovered - predicted_solution[:, 4])**2) / len(recovered)
        SSQ = np.sum((quarantined - predicted_solution[:, 3])**2) / len(quarantined)

        loss = SSI + SSR + SSQ
        return loss
    result = minimize(objective_function, coef_guess, args=(time, confirmed, quarantined, recovered), bounds=bounds)
    # result = differential_evolution(objective_function, bounds, args=(time, confirmed, recovered, pop))
    optimized_params = result.x

    return optimized_params.tolist()
def SEIQDRP_fit(data, pop):

    time = data['Date'].to_numpy()
    time = [i for i in range(len(time))]
    confirmed = data['Total_Current_Positive_Cases'].to_numpy() / pop
    recovered = data['Recovered'].to_numpy() / pop
    quarantined = data['Home_Confined'].to_numpy() / pop
    dead = data['Dead'].to_numpy() / pop

    coef_guess = [0.05018199919915549, 0.36119755018385835, 0.7085130515276519, 0.039401011876595976, 0.04084461787800722, 0.2823836409399339, 13.08872540261323, 0.020156623608517117, 0.0, 4.362924640953117]
    bounds = [
        (0,1),
        (0,5),
        (0,1),
        (0,1),
        (0,1),
        (0,1),
        (0,100),
        (0,1),
        (0,1),
        (0,100)
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
        predicted_solution = solve_ode_SEIQDRP(time_points, ini_values + params)

        SSQ = np.sum((quarantined - predicted_solution[:, 3]) ** 2)
        SSR = np.sum((recovered - predicted_solution[:, 4]) ** 2)
        SSD = np.sum((dead - predicted_solution[:, 5]) ** 2)
        SSI = np.sum((confirmed - predicted_solution[:, 2]) ** 2)
        loss = SSQ + SSR + SSD + SSI
        #print('loss:', loss, SSQ, SSR, SSD)
        return loss

    result = minimize(objective_function, coef_guess, args=(time, confirmed, quarantined, recovered, dead), bounds=bounds)
    optimized_params = result.x

    return optimized_params.tolist()
