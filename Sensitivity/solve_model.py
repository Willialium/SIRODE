from scipy.integrate import odeint
import numpy as np


def solve_ode_SEIR(t, params):
    def odes(y, t, beta, sigma, gamma):
        S, E, I, R = y

        dydt = [
            -beta * S * I,  # Susceptible cases
            beta * S * I - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I,  # Recovered cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3]]
    sol = odeint(odes, y, t, (params[4], params[5], params[6]))

    return sol


def solve_ode_SEIRP(t, params):
    def odes(y, t, alpha, beta, sigma, gamma):
        S, E, I, R, P = y

        dydt = [
            -alpha * S - beta * S * I,  # Susceptible cases
            beta * S * I - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I,  # Recovered cases
            alpha * S  # Insusceptible cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4]]
    sol = odeint(odes, y, t, (params[5], params[6], params[7], params[8]))

    return sol


def solve_ode_SEIQR(t, params):
    def odes(y, t, beta, sigma, gamma, laambda):
        S, E, I, Q, R = y

        dydt = [
            -beta * S * I,  # Susceptible cases
            beta * S * I - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I - laambda * Q,  # Quarantined cases
            laambda * Q,  # Recovered cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4]]
    sol = odeint(odes, y, t, (params[5], params[6], params[7], params[8]))

    return sol


def solve_ode_SEIQDRP(t, params):
    # print('solving with:', params)
    def odes(y, t, alpha, beta, sigma, gamma, l1, l2, l3, k1, k2, k3):
        S, E, I, Q, R, D, P = y

        def Lambda(t, l1, l2, l3):
            # return l1 / (1 + np.exp(-l2 * (t - l3)))
            return l1 + np.exp(-l2 * (t + l3))

        def Kappa(t, k1, k2, k3):
            return k1 / (np.exp(k2 * (t - k3)) + np.exp(-k2 * (t - k3)))
            # return k1*(np.exp(-(k2*(t-k3))**2))
            # return k1 + np.exp(-k2*(t+k3))

        dydt = [
            -alpha * S - beta * S * I,  # Susceptible cases
            beta * S * I - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I - Lambda(t, l1, l2, l3) * Q - Kappa(t, k1, k2, k3) * Q,  # Quarantined cases
            Lambda(t, l1, l2, l3) * Q,  # Recovered cases
            Kappa(t, k1, k2, k3) * Q,  # Dead cases
            alpha * S  # Insusceptible cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4], params[5], params[6]]
    sol = odeint(odes, y, t, (
    params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16]))
    return sol