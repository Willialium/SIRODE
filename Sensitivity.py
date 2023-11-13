import random as rnd
import copy

from solve_model import solve_ode


def get_sample_matrix(num_samples, params, param_bounds):
    result = []
    for j in range(len(params)):
        these_samples = []
        for i in range(num_samples):
            num = rnd.random()
            num = num * param_bounds[j] * params[j] * 2
            num = num + params[j] - param_bounds[j] * params[j]
            these_samples = these_samples + [num]
        result = result + [these_samples]
    return result


def get_AB(XA, XB, param_index, multiple=False):
    if(not multiple):
        XAB = copy.copy(XB)
        XAB[param_index] = XA[param_index]
    else:
        XAB = copy.copy(XB)
        for i in param_index:
            XAB[i] = XA[i]
    return XAB


def get_fun_list(param_matrix, t, N):
    res = []
    for i in range(len(param_matrix[0])):
        params = []
        for j in range(len(param_matrix)):
            params = params + [param_matrix[j][i]]
        sol = solve_ode(t, params + [N])

        val = sol[-1, 3] + sol[-1, 4] + sol[-1, 5]
        res += [val]
    return res


def calc_total_sobol(yA, yB, yAB):
    num1 = len(yA) * sum([yB[i] * yAB[i] for i in range(len(yA))])
    num2 = (sum(yA)) ** 2

    den1 = len(yA) * sum([i ** 2 for i in yA])
    den2 = num2

    tot = 1 - ((num1 - num2) / (den1 - den2))
    return tot


def get_total_indicies(num_samples, t, raw, params, param_bounds, N):
    sobols = []

    for i in range(len(params)):
        if (param_bounds[i] == 0):
            continue
        XA = get_sample_matrix(num_samples, params, param_bounds)
        XB = get_sample_matrix(num_samples, params, param_bounds)
        XAB = get_AB(XA, XB, i)
        print('Got sample matricies for param', i)
        fA = get_fun_list(XA, t, N)
        fB = get_fun_list(XB, t, N)
        fAB = get_fun_list(XAB, t, N)
        print('Got function values for param', i)
        calc = calc_total_sobol(fA, fB, fAB)
        print('sobol for param', i, calc)
        sobols += [calc]
    '''for i in range(raw[0]):
        Q0 = raw[2][i]
        I0 = .3 * Q0
        E0 = .3 * Q0
        R0 = raw[3][i]
        D0 = raw[3][i]
        S0 = N - Q0 - E0 - R0 - D0 - I0
        ini = [S0, E0, I0, Q0, R0, D0, 0]
        params = ini + params[7:]
        param_bounds = [0 for i in range(len(ini))] + params[len(ini):]
        print(params)
        print(param_bounds)

        XA = get_sample_matrix(num_samples, params, param_bounds)
        XB = get_sample_matrix(num_samples, params, param_bounds)
        XAB = get_AB(XA, XB, [i for i in range(len(ini))])
        print('Got sample matricies for time', i)
        fA = get_fun_list(XA, t, N)
        fB = get_fun_list(XB, t, N)
        fAB = get_fun_list(XAB, t, N)
        print('Got function values for time', i)
        calc = calc_total_sobol(fA, fB, fAB)
        print('sobol for time', i, calc)
        sobols += [calc]'''

    return sobols
