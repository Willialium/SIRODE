import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP
from tuneModel import SEIR_fit, SEIRP_fit, SEIQR_fit, SEIQDRP_fit

data = pd.read_csv('../combined_data.csv')[['Date', 'Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead']]
data = data[data['Date'] >= '2020-03-09']
data = data[data['Date'] <= '2020-06-01']

data = data.groupby('Date').sum().reset_index()
N = 6e7
confirmed = data['Total_Current_Positive_Cases'].to_numpy()
recovered = data['Recovered'].to_numpy()
quarantined = data['Home_Confined'].to_numpy()
dead = data['Dead'].to_numpy()
real_time = data['Date'].to_numpy()
time = [i for i in range(len(real_time))]
time_mult = 2
def SEIR_test():

    N=6e5

    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    S0 = N - E0 - R0 - I0
    ini = [S0, E0, I0, R0]

    fit_params = SEIR_fit(data, N)
    time2 = [t for t in range(len(time * time_mult))]

    ode_data = solve_ode_SEIR(time2, ini + fit_params + [N])


    def plot():
        plt.figure(figsize=(20,20))
        plt.subplot(2, 2, 1)
        plt.title('Population = {:.0e}\nSEIR Model with beta = {:.5f}, sigma = {:.5f}, gamma = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real')
        plt.plot(time2, ode_data[:, 3], label='Recovered')
        plt.scatter(time, confirmed, s=.4, label='Infected_real')
        plt.plot(time2, ode_data[:, 2], label='Infected')
        plt.plot(time2, ode_data[:, 1], label='Exposed')

        plt.subplots_adjust(hspace=.4)

        plt.legend()


    plot()
def SEIRP_test():

    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    S0 = N - E0 - R0 - I0
    ini = [S0, E0, I0, R0, 1]

    fit_params = SEIRP_fit(data, N)
    time2 = [t for t in range(len(time * time_mult))]

    ode_data = solve_ode_SEIRP(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 2)

        plt.title('Population = {:.0e}\nSEIRP Model with beta = {:.5f}, sigma = {:.5f}, gamma = {:.5f}, alpha = {:.5f}'.format(N, fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real')
        plt.plot(time2, ode_data[:, 3], label='Recovered')
        plt.scatter(time, confirmed, s=.4, label='Infected_real')
        plt.plot(time2, ode_data[:, 2], label='Infected')
        plt.plot(time2, ode_data[:, 1], label='Exposed')
        #plt.plot(time2, ode_data[:, 4], label='Immune')
        #plt.plot(time2, ode_data[:, 0], label='Susceptible')

        plt.legend()


    plot()
def SEIQR_test():
    N=6e5
    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    Q0 = quarantined[0]
    S0 = N - E0 - R0 - I0 - Q0
    ini = [S0, E0, I0, Q0, R0]

    fit_params = SEIQR_fit(data, N)
    time2 = [t for t in range(len(time*time_mult))]

    ode_data = solve_ode_SEIQR(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 3)

        plt.title('Population = {:.0e}\nSEIQR Model with beta = {:.5f}, sigma = {:.5f}, gamma = {:.5f}, delta = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real')
        plt.plot(time2, ode_data[:, 4], label='Recovered')
        plt.scatter(time, confirmed, s=.4, label='Infected_real')
        plt.plot(time2, ode_data[:, 2], label='Infected')
        plt.scatter(time, quarantined, s=.4, label='Quarantined_real')
        plt.plot(time2, ode_data[:, 3], label='Quarantined')
        plt.plot(time2, ode_data[:, 1], label='Exposed')


        plt.legend()


    plot()

def SEIQDRP_test():
    Q0 = quarantined[0]
    I0 = confirmed[0]
    E0 = confirmed[0]
    R0 = recovered[0]
    D0 = dead[0]
    S0 = N - Q0 - E0 - R0 - D0 - I0
    ini = [S0, E0, I0, Q0, R0, D0, 0]

    fit_params = SEIQDRP_fit(data, N)
    time2 = [t for t in range(len(time*time_mult))]

    ode_data = solve_ode_SEIQDRP(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 4)

        plt.title('Population = {:.0e}\nSEIQDRP Model with beta = {:.5f}, sigma = {:.5f}, gamma = {:.5f}, delta = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real')
        plt.plot(time2, ode_data[:, 4], label='Recovered')
        plt.scatter(time, confirmed, s=.4, label='Infected_real')
        plt.plot(time2, ode_data[:, 2], label='Infected')
        plt.scatter(time, quarantined, s=.4, label='Quarantined_real')
        plt.plot(time2, ode_data[:, 3], label='Quarantined')
        plt.plot(time2, ode_data[:, 1], label='Exposed')
        plt.plot(time2, ode_data[:, 5], label='Dead')

        plt.legend()


    plot()



SEIR_test()
SEIRP_test()
SEIQR_test()
SEIQDRP_test()
plt.show()