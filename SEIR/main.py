from fit_model import get_coef
from solve_model import solve_ode
import matplotlib.pyplot as plt
from scipy.stats import sobol_indices, uniform
from scipy.stats import linregress
import numpy as np
from tuneModel import workit
import pandas as pd

data = pd.read_csv('../combined_data.csv')[['Date', 'Total_Current_Positive_Cases', 'Recovered']]
data = data[data['Date'] >= '2020-03-09']
data = data[data['Date'] <= '2020-06-21']

data = data.groupby('Date').sum().reset_index()
N = 6e5
confirmed = data['Total_Current_Positive_Cases'].to_numpy()
recovered = data['Recovered'].to_numpy()
time = data['Date'].to_numpy()
time = [i for i in range(len(time))]

I0 = confirmed[0]
E0 = .3*I0
R0 = recovered[0]
S0 = N - E0 - R0 - I0
ini = [S0, E0, I0, R0]
#ini = [299986037.0, 2395.5, 2395.5, 724]

fit_params = workit(data, N)
print('Beta', fit_params[0], 'Sigma', fit_params[1], 'Gamma', fit_params[2])
time2 = [t / (6.0) for t in range(len(time * 60))]

ode_data = solve_ode(time, ini + fit_params + [N])


def plot():
    plt.figure(figsize=(12,5))
    # Label each curve
    # Calculate R2 for recovered and ode_data[:, 3]
    MSE_recovered = np.sum((recovered - ode_data[:, 3])**2) / len(recovered)
    MSE_infected = np.sum((confirmed - ode_data[:, 2])**2) / len(confirmed)
    #Print in scientific notation
    plt.title(f'MSE: Recovered: {MSE_recovered:.2e}\nMSE Infected: {MSE_infected:.2e}')

    plt.scatter(time, recovered, s=.4, label='Recovered_real')
    plt.plot(time, ode_data[:, 3], label='Recovered')
    plt.scatter(time, confirmed, s=.4, label='Infected_real')
    plt.plot(time, ode_data[:, 2], label='Infected')
    plt.plot(time, ode_data[:, 1], label='Exposed')

    plt.legend()
    plt.show()

plot()
