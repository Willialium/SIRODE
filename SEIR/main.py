import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solve_model import solve_ode_SEIR, solve_ode_SEIRP, solve_ode_SEIQR, solve_ode_SEIQDRP
from tuneModel import SEIR_fit, SEIRP_fit, SEIQR_fit, SEIQDRP_fit

data = pd.read_csv('../combined_data.csv')[['Date', 'Total_Current_Positive_Cases', 'Recovered', 'Home_Confined', 'Dead', 'Region_Name']]
#data = data[data['Region_Name'] == 'Lombardia']
#data = data.drop(columns=['Region_Name'])
data = data.groupby('Date').sum().reset_index()
def plot():
    plt.figure(figsize=(15,8))
    plot_data = data[data['Date'] >= '2020-03-01']
    plot_data = plot_data[plot_data['Date'] <= '2021-06-21']

    plot_real_time = plot_data['Date'].to_numpy()
    plot_time = [i for i in range(len(plot_real_time))]
    plt.plot(plot_time, plot_data['Total_Current_Positive_Cases'], label='Confirmed')
    plt.plot(plot_time, plot_data['Recovered'], label='Recovered')
    plt.plot(plot_time, plot_data['Home_Confined'], label='Quarantined')
    plt.xticks(plot_time[::30], plot_real_time[::30], rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.show()

data = data[data['Date'] >= '2020-03-01']
data = data[data['Date'] <= '2020-06-20']

N = 6e7
confirmed = data['Total_Current_Positive_Cases'].to_numpy()
recovered = data['Recovered'].to_numpy()
quarantined = data['Home_Confined'].to_numpy()
dead = data['Dead'].to_numpy()
real_time = data['Date'].to_numpy()
time = [i for i in range(len(real_time))]
time_mult = 1

def SEIR_test():

    #N=6e5

    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    S0 = N - E0 - R0 - I0
    ini = [S0, E0, I0, R0]

    fit_params = SEIR_fit(data, N)
    print(fit_params)
    time2 = [t for t in range(len(time * time_mult))]

    ode_data = solve_ode_SEIR(time2, ini + fit_params + [N])


    def plot():
        plt.figure(figsize=(15,8))
        plt.subplots_adjust(hspace=.4)
        plt.subplot(2, 2, 1)
        plt.title('Population = {:.0e}\nSEIR Model with beta = {:.5f}\nsigma = {:.5f}, gamma = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real', color='blue')
        plt.plot(time2, ode_data[:, 3], label='Recovered', color='blue')
        plt.scatter(time, confirmed, s=.4, label='Infected_real', color='orange')
        plt.plot(time2, ode_data[:, 2], label='Infected', color='orange')
        plt.plot(time2, ode_data[:, 1], label='Exposed', color='green')
        #plt.plot(time2, ode_data[:, 0], label='Susceptible')

        ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta\frac{S*I}{N}$'+ '\n' + \
                    r'$\frac{dE}{dt} = \beta\frac{S*I}{N} - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I$'
        plt.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt.legend()


    plot()
def SEIRP_test():

    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    S0 = N - E0 - R0 - I0
    ini = [S0, E0, I0, R0, 1]

    fit_params = SEIRP_fit(data, N)
    print(fit_params)
    time2 = [t for t in range(len(time * time_mult))]

    ode_data = solve_ode_SEIRP(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 2)

        plt.title('Population = {:.0e}\nSEIRP Model with beta = {:.5f}, sigma = {:.5f}\ngamma = {:.5f}, alpha = {:.5f}'.format(N, fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real', color='blue')
        plt.plot(time2, ode_data[:, 3], label='Recovered', color='blue')
        plt.scatter(time, confirmed, s=.4, label='Infected_real', color='orange')
        plt.plot(time2, ode_data[:, 2], label='Infected', color='orange')
        plt.plot(time2, ode_data[:, 1], label='Exposed', color='green')
        #plt.plot(time2, ode_data[:, 4], label='Immune')
        #plt.plot(time2, ode_data[:, 0], label='Susceptible')

        ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta\frac{S*I}{N}$'+ '\n' + \
                    r'$\frac{dE}{dt} = \beta\frac{S*I}{N} - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I$' + '\n' + \
                    r'$\frac{dP}{dt} = \alpha S$'
        plt.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt.legend()


    plot()
def SEIQR_test():
    N=6e7
    I0 = confirmed[0]
    E0 = I0
    R0 = recovered[0]
    Q0 = quarantined[0]
    S0 = N - E0 - R0 - I0 - Q0
    ini = [S0, E0, I0, Q0, R0]

    fit_params = SEIQR_fit(data, N)
    print(fit_params)
    time2 = [t for t in range(len(time*time_mult))]

    ode_data = solve_ode_SEIQR(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 3)

        plt.title('Population = {:.0e}\nSEIQR Model with beta = {:.5f}, sigma = {:.5f}\ngamma = {:.5f}, delta = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real', color='blue')
        plt.plot(time2, ode_data[:, 4], label='Recovered', color='blue')
        plt.scatter(time, confirmed, s=.4, label='Infected_real', color='orange')
        plt.plot(time2, ode_data[:, 2], label='Infected', color='orange')
        plt.scatter(time, quarantined, s=.4, label='Quarantined_real', color='red')
        plt.plot(time2, ode_data[:, 3], label='Quarantined', color='red')
        #plt.plot(time2, ode_data[:, 1], label='Exposed', color='green')

        ode_latex = r'$\frac{dS}{dt} = -\beta\frac{S*I}{N}$'+ '\n' + \
                    r'$\frac{dE}{dt} = \beta\frac{S*I}{N} - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \delta I$' + '\n' + \
                    r'$\frac{dQ}{dt} = \delta I - \gamma Q$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma Q$'
        plt.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

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
    print(fit_params)
    time2 = [t for t in range(len(time*time_mult))]

    ode_data = solve_ode_SEIQDRP(time2, ini + fit_params + [N])


    def plot():
        plt.subplot(2, 2, 4)

        plt.title('Population = {:.0e}\nSEIQDRP Model with alpha = {:.5f}, beta = {:.5f}\nsigma = {:.5f}, gamma = {:.5f}'.format(N,fit_params[0], fit_params[1], fit_params[2], fit_params[3]))

        plt.ylabel('Population in Compartment')
        #plt.xticks(np.arange(0, len(real_time), len(real_time)-1), [''] + [real_time[-1]], rotation=90)
        plt.xticks(np.arange(0, len(real_time), 5), real_time[::5], rotation=90)

        plt.scatter(time, recovered, s=.4, label='Recovered_real', color='blue')
        plt.plot(time2, ode_data[:, 4], label='Recovered', color='blue')
        plt.scatter(time, confirmed, s=.4, label='Infected_real', color='orange')
        plt.plot(time2, ode_data[:, 2], label='Infected', color='orange')
        plt.scatter(time, quarantined, s=.4, label='Quarantined_real', color='red')
        plt.plot(time2, ode_data[:, 3], label='Quarantined', color='red')
        plt.plot(time2, ode_data[:, 1], label='Exposed', color='green')
        plt.plot(time2, ode_data[:, 5], label='Dead', color='black')

        ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta\frac{S*I}{N}$'+ '\n' + \
                    r'$\frac{dE}{dt} = \beta\frac{S*I}{N} - \gamma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \gamma E - \delta I$' + '\n' + \
                    r'$\frac{dQ}{dt} = \delta I - \lambda(t) Q$ - ' + r'$\kappa(t) Q$' + '\n' + \
                    r'$\frac{dR}{dt} = \lambda(t) Q$' + '\n' + \
                    r'$\frac{dD}{dt} = \kappa(t) Q$' + '\n' + \
                    r'$\frac{dP}{dt} = \alpha S$'
        plt.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt.legend()


    plot()




def test_ode():
    SEIR_test()
    SEIRP_test()
    SEIQR_test()
    SEIQDRP_test()

    plt.tight_layout()
    plt.show()

#plot()
test_ode()