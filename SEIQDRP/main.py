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
Q0 = quarantined[0]
I0 = .3 * Q0
E0 = .3 * Q0
R0 = recovered[0]
D0 = dead[0]
S0 = N - Q0 - E0 - R0 - D0 - I0
ini = [S0, E0, I0, Q0, R0, D0, 0]
print(ini)
#fit_params = workit(data, N)
fit_params = [0.009429856502123, 1.190550770326202, 0.996835367442656,
              0.999998889756488, 0.016344773987195, 0.303411607828428, 14.886187811910293,
              0.031761601509612, 0.059964445084184, 4.735594221653220]
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
    plt.scatter(time, recovered, s=.4, label='Recovered_real')
    plt.scatter(time, dead, s=.4)
    plt.plot(time2, ode_data[:, 3])
    plt.plot(time2, ode_data[:, 4])
    plt.plot(time2, ode_data[:, 5])
    plt.legend()
    plt.show()

plot()

#sobols = get_total_indicies(1000000, time, [quarantined, recovered, dead], fit_params, [7, .1, .1, .3, .101010, 0, 0, 0, 0, 0, 0], N)
#print(sobols)

#plot()
