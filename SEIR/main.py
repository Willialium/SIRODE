from fit_model import get_coef
from solve_model import solve_ode
import matplotlib.pyplot as plt
from scipy.stats import sobol_indices, uniform
from scipy.stats import linregress
import numpy as np
from tuneModel import workit
import pandas as pd

data = pd.read_csv('../combined_data.csv')[['Date', 'Total_Current_Positive_Cases', 'Recovered']]
data = data[data['Date'] < '2020-04-01']
data = data.groupby('Date').sum().reset_index()
print(data)

N = 6e7
confirmed = data['Total_Current_Positive_Cases'].to_numpy()
recovered = data['Recovered'].to_numpy()
time = data['Date'].to_numpy()
time = [i for i in range(len(time))]

I0 = confirmed[0]
E0 = .3*I0
R0 = recovered[0]
S0 = N - E0 - R0 - I0
ini = [S0, E0, I0, R0]

fit_params = workit(data, N)
print('Beta', fit_params[0], 'Sigma', fit_params[1], 'Gamma', fit_params[2])
time2 = [t / (6.0) for t in range(len(time * 6))]
ode_data = solve_ode(time2, ini + fit_params + [N])


def plot():
    '''tot_i = confirmed.tolist() + recovered.tolist() + quarantined.tolist()
    tot_f = ode_data[:, 3].tolist() + ode_data[:, 4].tolist() + ode_data[:, 5].tolist()
    tot_f = [tot_f[i*6] for i in range(len(tot_i))]
    lin = linregress(tot_i, tot_f)
    print('R2:', lin.rvalue)'''

    plt.figure(figsize=(4,5))
    plt.scatter(time, recovered, s=.4)
    plt.scatter(time, confirmed, s=.4)
    #plt.plot(time2, ode_data[:, 0])
    plt.plot(time2, ode_data[:, 1])
    plt.plot(time2, ode_data[:, 2])
    plt.plot(time2, ode_data[:, 3])

    plt.show()

plot()
