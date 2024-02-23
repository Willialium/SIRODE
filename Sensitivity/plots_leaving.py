import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from models import SIS_test, SIR_test, SEI_test, SEIR_test, SEIRP_test, data
from matplotlib.widgets import Slider, Button
from models import scales, inis, params, time_interval

EXCLUDE = 'S'
base = 5
scales['time'] = 4

global use_SEIQRDP
use_SEIQRDP = False

# utility functions
def create_slider(ax, label, valmin, valmax, valinit, callback):
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
    slider.on_changed(callback)
    return slider
def find_intersection(str1, str2):
    return "".join(set(str1).intersection(set(str2)))

#############################
##### UPDATE FUNCTIONS  #####
#############################

def update_SIR(x, update=True):
    scales['SIR']['Xbeta'] = base**SIR_betaSlider.val
    scales['SIR']['Xgamma'] = base**SIR_gammaSlider.val
    SIR_Update = SIR_test(use_SEIQRDP)
    model_buffer = 0

    for i in range(len('SIR') - len(find_intersection('SIR', EXCLUDE))):
        while 'SIR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SIR_lines[i][0].set_ydata(SIR_Update[i+model_buffer])
        SIR_lines[i][0].set_xdata(np.arange(0, len(SIR_Update[i+model_buffer])))
        if 'SIR'[i+model_buffer] == 'S':
            continue
        d_SIR_lines[i][0].set_ydata(np.diff(SIR_Update[i+model_buffer]))
        d_SIR_lines[i][0].set_xdata(np.arange(0, len(SIR_Update[i+model_buffer])-1))

    ax1.relim()
    ax2.relim()
    ax1.autoscale_view()
    ax2.autoscale_view()
    ax1.title.set_text(f'beta = {params["SIR"]["beta"]*scales["SIR"]["Xbeta"]:.2f}, gamma = {params["SIR"]["gamma"]*scales["SIR"]["Xgamma"]:.2f}')
    fig.canvas.draw_idle()

def update_all(x):
    inis['R0'] = R0_slider.val
    inis['I0'] = I0_slider.val
    scales['time'] = int(time_slider.val)

    update_SIR(x, update=False)

def button_press(x):
    global use_SEIQRDP
    use_SEIQRDP = not use_SEIQRDP
    if use_SEIQRDP:
        button.label.set_text('Use fitted\ndata')
    else:
        button.label.set_text('Use SEIQRDP')
    update_all(x)


########################
##### CREATE PLOTS #####
########################
'''def create_real_plot(ax, name):
    time = np.arange(0, len(data))
    for cat in name:
        if cat == 'I':
            ax.scatter(time, data['Total_Current_Positive_Cases'], s=.4, color='red')
        elif cat == 'R':
            ax.scatter(time, data['Recovered'], s=.4, color='orange')
        elif cat == 'D':
            ax.scatter(time, data['Dead'], s=.4, color='black')
        elif cat == 'Q':
            ax.scatter(time, data['Home_Confined'], s=.4, color='blue')
'''

def create_SIR_plot(ax, SIR_test_result):
    lines = []
    colors = ['yellow', 'red', 'orange']
    for i in range(3):
        if 'SIR'[i] not in EXCLUDE:
            lines.append(ax.plot(SIR_test_result[i], color=colors[i], label='SIR'[i]))
    ax.title.set_text(f'beta = {params["SIR"]["beta"]*scales["SIR"]["Xbeta"]:.2f}, gamma = {params["SIR"]["gamma"]*scales["SIR"]["Xgamma"]:.2f}')
    ax.set_ylabel('Proportion in\neach compartment')
    ax.legend()

    ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                r'$\frac{dI}{dt} = \beta SI$ - ' + r'$\gamma I$' + '\n'  \
                r'$\frac{dR}{dt} = \gamma I$'

    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')

    return lines
def create_d_plot(ax, SIR_test_result):

    lines = []
    colors = ['yellow', 'red', 'orange']
    dSdt = np.diff(SIR_test_result[0])
    dIdt = np.diff(SIR_test_result[1])
    dRdt = np.diff(SIR_test_result[2])
    datas = [dSdt, dIdt, dRdt]
    for i in range(3):
        if 'SIR'[i] not in EXCLUDE:
            lines.append(ax.plot(datas[i], color=colors[i], label='d'+'SIR'[i]+'/dt'))


    ax.legend()

    return lines


#########################
##### CREATE FIGURE #####
#########################
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 10), sharex=True)


SIR_lines = create_SIR_plot(ax1, SIR_test(use_SEIQRDP))
d_SIR_lines = create_d_plot(ax2, SIR_test(use_SEIQRDP))

##########################
##### CREATE SLIDERS #####
##########################
plt.subplots_adjust(top=.85, bottom=.15, wspace=.32)

SIR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.08, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SIR)
SIR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.05, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SIR)

R0_slider = create_slider(ax=fig.add_axes([0.1, 0.97, 0.7, 0.03]), label='R0', valmin=0.0000001, valmax=0.03, valinit=inis['R0'], callback=update_all)
I0_slider = create_slider(ax=fig.add_axes([0.1, 0.94, 0.7, 0.03]), label='I0', valmin=0.0, valmax=.0001, valinit=inis['I0'], callback=update_all)
time_slider = create_slider(ax=fig.add_axes([0.1, 0.91, 0.7, 0.03]), label='time', valmin=1, valmax=10, valinit=scales['time'], callback=update_all)

button = Button(fig.add_axes([0.9, 0.9, 0.085, 0.085]), 'Use SEIQRDP\nparams', color='lightgrey', hovercolor='0.9')
button.on_clicked(button_press)

fig.canvas.manager.full_screen_toggle()
plt.show()
