import get_country_data
from fit_model import get_coef
from solve_model import solve_ode
import matplotlib.pyplot as plt
from scipy.stats import sobol_indices, uniform
from scipy.stats import linregress
import numpy as np
from tuneModel import workit
from Sensitivity import get_total_indicies

data = get_country_data.get_cleaned_country_data('Italy', '2020-03-09', '2020-04-20')
N = 3e8
confirmed = data['Confirmed'].to_numpy()
recovered = data['Recovered'].to_numpy()
dead = data['Dead'].to_numpy()
time = data['Date'].to_numpy()
time = [i for i in range(len(time))]

quarantined = confirmed - recovered - dead
I0 = .3 * Q0
E0 = .3 * Q0
R0 = recovered[0]
S0 = N - Q0 - E0 - R0 - D0 - I0
ini = [S0, E0, I0, R0]
#fit_params = workit(data, N)
print(fit_params)
time2 = [t / (6.0) for t in range(len(time * 6))]
ode_data = solve_ode(time2, ini + fit_params + [N])


def plot():
    tot_i = quarantined.tolist() + recovered.tolist() + dead.tolist()
    tot_f = ode_data[:, 3].tolist() + ode_data[:, 4].tolist() + ode_data[:, 5].tolist()
    tot_f = [tot_f[i*6] for i in range(len(tot_i))]
    lin = linregress(tot_i, tot_f)
    print('R2:', lin.rvalue)

    plt.figure(figsize=(6,5))
    plt.scatter(time, quarantined, s=.4)
    plt.scatter(time, recovered, s=.4)
    plt.scatter(time, dead, s=.4)
    plt.plot(time2, ode_data[:, 3])
    plt.plot(time2, ode_data[:, 4])
    plt.plot(time2, ode_data[:, 5])

    plt.show()

#plot()


