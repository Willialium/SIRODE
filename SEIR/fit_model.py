import numpy as np

def get_coef():

    numdays = 42

    beta = 1.19055077
    delta = .9999999
    gamma = .996835367
    N = 3e8

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
