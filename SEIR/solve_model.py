from scipy.integrate import odeint
import numpy as np


def solve_ode_SEIR(t, params):
    def odes(y, t, beta, sigma, gamma, N):
        S, E, I, R = y

        dydt = [
            -beta * S * I / N,  # Susceptible cases
            beta * S * I / N - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I,  # Recovered cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3]]
    sol = odeint(odes, y, t, (params[4], params[5], params[6], params[7]))

    return sol


def solve_ode_SEIRP(t, params):
    def odes(y, t, alpha, beta, sigma, gamma, N):
        S, E, I, R, P = y

        dydt = [
            -alpha * S - beta * S * I / N,  # Susceptible cases
            beta * S * I / N - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I,  # Recovered cases
            alpha * S  # Insusceptible cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4]]
    sol = odeint(odes, y, t, (params[5], params[6], params[7], params[8], params[9]))

    return sol


def solve_ode_SEIQR(t, params):
    def odes(y, t, beta, sigma, gamma, delta, N):
        S, E, I, Q, R = y

        dydt = [
            -beta * S * I / N,  # Susceptible cases
            beta * S * I / N - sigma * E,  # Exposed cases
            sigma * E - delta * I,  # Infectious cases
            delta * I - gamma * Q,  # Quarantined cases
            gamma * Q,  # Recovered cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4]]
    sol = odeint(odes, y, t, (params[5], params[6], params[7], params[8], params[9]))

    return sol


def solve_ode_SEIQDRP(t, params):
    # print('solving with:', params)
    def odes(y, t, alpha, beta, sigma, gamma, l1, l2, l3, k1, k2, k3, N):
        S, E, I, Q, R, D, P = y

        def Lambda(t, l1, l2, l3):
            # return l1 / (1 + np.exp(-l2 * (t - l3)))
            return l1 + np.exp(-l2 * (t + l3))

        def Kappa(t, k1, k2, k3):
            return k1 / (np.exp(k2 * (t - k3)) + np.exp(-k2 * (t - k3)))
            # return k1*(np.exp(-(k2*(t-k3))**2))
            # return k1 + np.exp(-k2*(t+k3))

        dydt = [
            -alpha * S - beta * S * I / N,  # Susceptible cases
            beta * S * I / N - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gamma * I - Lambda(t, l1, l2, l3) * Q - Kappa(t, k1, k2, k3) * Q,  # Quarantined cases
            Lambda(t, l1, l2, l3) * Q,  # Recovered cases
            Kappa(t, k1, k2, k3) * Q,  # Dead cases
            alpha * S  # Insusceptible cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4], params[5], params[6]]
    sol = odeint(odes, y, t, (
    params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16],
    params[17]))

    return sol


def solve_ode_SEIR_2(t, params):
    def odes(y, t, tau, beta1, beta2, mu, kappa, gamma, mu1, delta, alpha, nu, N):
        S, E, I, R = y

        dydt = [
            tau - (mu + alpha) * S - (beta1 * E + beta2 * I) * S,  # Susceptible cases
            (beta1 * E + beta2 * I) * S - (kappa + mu + gamma) * E,  # Exposed cases
            kappa * E - (mu + mu1 + delta + nu) * I,  # Infectious cases
            delta * I - mu * R,  # Recovered cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3]]
    sol = odeint(odes, y, t, (
    params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13],
    params[14]))

    return sol

def solve_ode_SEIRDC(t, params):
    def odes(y, t, beta0, alpha, sigma, gamma, gammaR, d, laambda, N):
        S, E, I, R, D, C = y

        def beta():
            return beta0 * (1 - alpha) * (1 - D / N) ** 1000


        dydt = [
            -beta() * S * I / N,  # Susceptible cases
            beta() * S * I / N - sigma * E,  # Exposed cases
            sigma * E - gamma * I,  # Infectious cases
            gammaR * I,  # Recovered cases
            d * gamma * I - laambda * D,  # awareness cases
            sigma * E  # Cumulative dead cases
        ]
        return dydt

    y = [params[0], params[1], params[2], params[3], params[4], params[5]]
    sol = odeint(odes, y, t, (params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13]))

    return sol