from scipy.integrate import odeint
import numpy as np


def solve_ode(t, params):
    def odes(y, t, beta, delta, gamma, N):

        S, E, I, R = y

        dydt = [
            -beta * S * I / N,  # Susceptible cases
            beta * S * I / N - delta * E,  # Exposed cases
            delta * E - gamma * I,  # Infectious cases
            gamma * I,  # Recovered cases
        ]
        return dydt
    y = [params[0], params[1], params[2], params[3]]
    sol = odeint(odes, y, t, (params[4], params[5], params[6], params[7]))

    return sol
