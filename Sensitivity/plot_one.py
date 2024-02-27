import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from models import scales, params, inis, data
from models import SIS_test, SIR_test, SEI_test, SEIR_test, SEIRP_test, SEIQR_test, SEIQRDP_test

show_data = False
which = 'SEIQRDP'
EXCLUDE = 'SP'
base = 3

def create_slider(ax, label, valmin, valmax, valinit, callback):
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
    slider.on_changed(callback)
    return slider
def find_intersection(str1, str2):
    return "".join(set(str1).intersection(set(str2)))

def update(x):
    for val in scales[which]:
        scales[which][val] = base**sliders[val].val

    update_vals = []


    if which == 'SIS':
        update_vals = SIS_test()
    elif which == 'SIR':
        update_vals = SIR_test()
    elif which == 'SEI':
        update_vals = SEI_test()
    elif which == 'SEIR':
        update_vals = SEIR_test()
    elif which == 'SEIRP':
        update_vals = SEIRP_test()
    elif which == 'SEIQR':
        update_vals = SEIQR_test()
    elif which == 'SEIQDRP':
        update_vals = SEIQRDP_test()

    model_buffer = 0

    for i in range(len(which if which != 'SIS' else 'SI') - len(find_intersection(which, EXCLUDE))):
        while which[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        lines[i][0].set_ydata(update[i+model_buffer])
        lines[i][0].set_xdata(np.arange(0, len(update[i+model_buffer])))

    ax.relim()
    ax.autoscale_view()
    ax.title.set_text(' '.join(set(f'{p}: {params[which][p]*scales[which][s]:.2f}' for p in params[which] for s in scales[which])))
    fig.canvas.draw_idle()

def create_real_plot():
    time = np.arange(0, len(data))
    for cat in which:
        if cat in EXCLUDE:
            continue
        if cat == 'I':
            ax.scatter(time, data['Total_Current_Positive_Cases'], s=.4, color='red')
        elif cat == 'R':
            ax.scatter(time, data['Recovered'], s=.4, color='orange')
        elif cat == 'D':
            ax.scatter(time, data['Dead'], s=.4, color='black')
        elif cat == 'Q':
            ax.scatter(time, data['Home_Confined'], s=.4, color='blue')
def create_plot(initial_lines):
    lines = []
    colors = {'S':'yellow', 'E':'green', 'I':'red', 'R':'orange', 'P':'purple', 'Q':'blue', 'D':'black'}
    for i in range(len(which if which != 'SIS' else 'SI')):
        if which[i] not in EXCLUDE:
            lines.append(ax.plot(initial_lines[i], color=colors[which[i]], label=which[i]))
    ax.title.set_text(' '.join(set(f'{p}: {params[which][p]*scales[which][s]:.2f}' for p in params[which] for s in scales[which])))
    ax.set_ylabel('Proportion in\neach compartment')
    ax.legend()

    if which == 'SIS':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dI}{dt} = \beta SI$' + '\n'
    elif which == 'SIR':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dI}{dt} = \beta SI - \gamma I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I$' + '\n'
    elif which == 'SEI':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E$' + '\n'
    elif which == 'SEIR':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I$' + '\n'
    elif which == 'SEIRP':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I - \rho R$' + '\n' + \
                    r'$\frac{dP}{dt} = \rho R$' + '\n'
    elif which == 'SEIQR':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I - \mu I$' + '\n' + \
                    r'$\frac{dQ}{dt} = \mu I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I$' + '\n'
    elif which == 'SEIQRDP':
        ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                    r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                    r'$\frac{dI}{dt} = \sigma E - \gamma I - \mu I$' + '\n' + \
                    r'$\frac{dQ}{dt} = \mu I$' + '\n' + \
                    r'$\frac{dR}{dt} = \gamma I - \rho R$' + '\n' + \
                    r'$\frac{dD}{dt} = \rho R$' + '\n' + \
                    r'$\frac{dP}{dt} = \rho R$' + '\n'


    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')

    return lines


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(right=.85, bottom=.23, top=.85)
sliders = {}

lines = []
if which == 'SIS':
    lines = create_plot(SIS_test())
elif which == 'SIR':
    lines = create_plot(SIR_test())
elif which == 'SEI':
    lines = create_plot(SEI_test())
elif which == 'SEIR':
    lines = create_plot(SEIR_test())
elif which == 'SEIRP':
    lines = create_plot(SEIRP_test())
elif which == 'SEIQR':
    lines = create_plot(SEIQR_test())
elif which == 'SEIQRDP':
    lines = create_plot(SEIQRDP_test())

initial_location = [0.1, 0.1, 0.8, 0.03]
for i in params[which]:
    sliders[i] = create_slider(fig.add_axes(initial_location), label=i, valmin=-2, valmax=2, valinit=0, callback=update)
    initial_location[1] -= 0.05

R0_slider = create_slider(ax=fig.add_axes([0.1, 0.97, 0.8, 0.03]), label='R0', valmin=0.0000001, valmax=0.03, valinit=inis['R0'], callback=update)
I0_slider = create_slider(ax=fig.add_axes([0.1, 0.94, 0.8, 0.03]), label='I0', valmin=0.0, valmax=.0001, valinit=inis['I0'], callback=update)
time_slider = create_slider(ax=fig.add_axes([0.1, 0.91, 0.8, 0.03]), label='time', valmin=1, valmax=10, valinit=scales['time'], callback=update)

plt.show()

