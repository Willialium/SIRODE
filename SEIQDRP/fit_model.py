import numpy as np

def get_coef():

    numdays = 42

    alpha = .009429856
    beta = 1.19055077
    gamma = .996835367
    delta = .9999999
    Lambda0 = [.01634477, 0.3034116]
    Kappa0 = [.0317616, .05996444]
    N = 3e8

    def Lambda(t, l1, l2, l3):
        return l1 / (1 + np.exp(-l2 * (t - l3)))

    def Kappa(t, k1, k2, k3):
        return k1 / np.exp(k2*(t - k3) + np.exp(-k2*(t - k3)))

    # Initial Guess
    y0 = [0,
          23955, # Exposed
          32395,
          7985,
          724,
          463,
          0]
    y0[0] = N - y0[3] - y0[1] - y0[4] - y0[5] - y0[2]

    t0 = [0]+[(i+1)/24 for i in range(numdays*24)]

    return [[alpha, beta, delta, gamma, Lambda, Kappa, N], y0, t0]
