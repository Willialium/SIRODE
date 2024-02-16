import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from models import SEIR_test, SEIRP_test, SEIQR_test, SEIQRDP_test, data
from matplotlib.widgets import Slider, Button
from models import scales, inis, params

EXCLUDE = 'SPR'
base = 2

global use_SEIQRDP
use_SEIQRDP = False

# utility functions
def create_slider(ax, label, valmin, valmax, valinit, callback):
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
    slider.on_changed(callback)
    return slider
def find_intersection(str1, str2):
    return "".join(set(str1).intersection(set(str2)))
def update_params():
    with open('params.txt', 'w') as f:

        temp_params = copy.deepcopy(params)
        temp_params['SEIR']['beta'] = temp_params['SEIR']['beta'] * scales['SEIR']['Xbeta']
        temp_params['SEIR']['sigma'] = temp_params['SEIR']['sigma'] * scales['SEIR']['Xsigma']
        temp_params['SEIR']['gamma'] = temp_params['SEIR']['gamma'] * scales['SEIR']['Xgamma']
        temp_params['SEIRP']['beta'] = temp_params['SEIRP']['beta'] * scales['SEIRP']['Xbeta']

        temp_params['SEIRP']['sigma'] = temp_params['SEIRP']['sigma'] * scales['SEIRP']['Xsigma']
        temp_params['SEIRP']['beta'] = temp_params['SEIRP']['beta'] * scales['SEIRP']['Xbeta']
        temp_params['SEIRP']['gamma'] = temp_params['SEIRP']['gamma'] * scales['SEIRP']['Xgamma']
        temp_params['SEIRP']['alpha'] = temp_params['SEIRP']['alpha'] * scales['SEIRP']['Xalpha']

        temp_params['SEIQR']['beta'] = temp_params['SEIQR']['beta'] * scales['SEIQR']['Xbeta']
        temp_params['SEIQR']['sigma'] = temp_params['SEIQR']['sigma'] * scales['SEIQR']['Xsigma']
        temp_params['SEIQR']['gamma'] = temp_params['SEIQR']['gamma'] * scales['SEIQR']['Xgamma']
        temp_params['SEIQR']['lambda'] = temp_params['SEIQR']['lambda'] * scales['SEIQR']['Xlambda']

        temp_params['SEIQRDP']['beta'] = temp_params['SEIQRDP']['beta'] * scales['SEIQRDP']['Xbeta']
        temp_params['SEIQRDP']['sigma'] = temp_params['SEIQRDP']['sigma'] * scales['SEIQRDP']['Xsigma']
        temp_params['SEIQRDP']['gamma'] = temp_params['SEIQRDP']['gamma'] * scales['SEIQRDP']['Xgamma']
        temp_params['SEIQRDP']['alpha'] = temp_params['SEIQRDP']['alpha'] * scales['SEIQRDP']['Xalpha']

        f.write(json.dumps(temp_params, indent=3))

#############################
##### UPDATE FUNCTIONS  #####
#############################

def update_SEIR(x, update=True):
    scales['SEIR']['Xbeta'] = base**SEIR_betaSlider.val
    scales['SEIR']['Xsigma'] = base**SEIR_sigmaSlider.val
    scales['SEIR']['Xgamma'] = base**SEIR_gammaSlider.val
    if update: update_params()

    SEIR_Update = SEIR_test(use_SEIQRDP)
    model_buffer = 0

    for i in range(len('SEIR') - len(find_intersection('SEIR', EXCLUDE))):
        while 'SEIR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIR_lines[i][0].set_ydata(SEIR_Update[i+model_buffer])
        SEIR_lines[i][0].set_xdata(np.arange(0, len(SEIR_Update[i+model_buffer])))
    ax1.relim()
    ax1.autoscale_view()
    ax1.title.set_text(f'beta = {params["SEIR"]["beta"]*scales["SEIR"]["Xbeta"]:.2f}, sigma = {params["SEIR"]["sigma"]*scales["SEIR"]["Xsigma"]:.2f}, gamma = {params["SEIR"]["gamma"]*scales["SEIR"]["Xgamma"]:.2f}')
    fig.canvas.draw_idle()
def update_SEIRP(x, update=True):
    scales['SEIRP']['Xbeta'] = base**SEIRP_betaSlider.val
    scales['SEIRP']['Xsigma'] = base**SEIRP_sigmaSlider.val
    scales['SEIRP']['Xgamma'] = base**SEIRP_gammaSlider.val
    scales['SEIRP']['Xalpha'] = base**SEIRP_alphaSlider.val
    if update: update_params()

    SEIRP_Update = SEIRP_test(use_SEIQRDP)
    model_buffer = 0
    for i in range(len('SEIRP') - len(find_intersection('SEIRP', EXCLUDE))):
        while 'SEIRP'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIRP_lines[i][0].set_ydata(SEIRP_Update[i+model_buffer])
        SEIRP_lines[i][0].set_xdata(np.arange(0, len(SEIRP_Update[i+model_buffer])))

    ax2.relim()
    ax2.autoscale_view()
    ax2.title.set_text(f'alpha = {params["SEIRP"]["alpha"]*scales["SEIRP"]["Xalpha"]:.2f}, beta = {params["SEIRP"]["beta"]*scales["SEIRP"]["Xbeta"]:.2f}, sigma = {params["SEIRP"]["sigma"]*scales["SEIRP"]["Xsigma"]:.2f}, gamma = {params["SEIRP"]["gamma"]*scales["SEIRP"]["Xgamma"]:.2f}')
    fig.canvas.draw_idle()
def update_SEIQR(x, update=True):
    scales['SEIQR']['Xbeta'] = base**SEIQR_betaSlider.val
    scales['SEIQR']['Xsigma'] = base**SEIQR_sigmaSlider.val
    scales['SEIQR']['Xgamma'] = base**SEIQR_gammaSlider.val
    scales['SEIQR']['Xlaambda'] = base**SEIQR_lambdaSlider.val
    if update: update_params()

    SEIQR_Update = SEIQR_test(use_SEIQRDP)
    model_buffer = 0
    for i in range(len('SEIQR') - len(find_intersection('SEIQR', EXCLUDE))):
        while 'SEIQR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIQR_lines[i][0].set_ydata(SEIQR_Update[i+model_buffer])
        SEIQR_lines[i][0].set_xdata(np.arange(0, len(SEIQR_Update[i+model_buffer])))

    ax3.relim()
    ax3.autoscale_view()
    ax3.title.set_text(f'beta = {params["SEIQR"]["beta"]*scales["SEIQR"]["Xbeta"]:.2f}, sigma = {params["SEIQR"]["sigma"]*scales["SEIQR"]["Xsigma"]:.2f}, gamma = {params["SEIQR"]["gamma"]*scales["SEIQR"]["Xgamma"]:.2f}, lambda = {params["SEIQR"]["lambda"]*scales["SEIQR"]["Xlambda"]:.2f}')
    fig.canvas.draw_idle()
def update_SEIQRDP(x, update=True):
    scales['SEIQRDP']['Xbeta'] = base**SEIQRDP_betaSlider.val
    scales['SEIQRDP']['Xsigma'] = base**SEIQRDP_sigmaSlider.val
    scales['SEIQRDP']['Xgamma'] = base**SEIQRDP_gammaSlider.val
    scales['SEIQRDP']['Xalpha'] = base**SEIQRDP_alphaSlider.val
    if update: update_params()

    SEIQRDP_Update = SEIQRDP_test()
    model_buffer = 0
    for i in range(len('SEIQRDP') - len(find_intersection('SEIQRDP', EXCLUDE))):
        while 'SEIQRDP'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIQRDP_lines[i][0].set_ydata(SEIQRDP_Update[i+model_buffer])
        SEIQRDP_lines[i][0].set_xdata(np.arange(0, len(SEIQRDP_Update[i+model_buffer])))

    ax4.title.set_text(f'alpha = {params["SEIQRDP"]["alpha"]*scales["SEIQRDP"]["Xalpha"]:.2f}, beta = {params["SEIQRDP"]["beta"]*scales["SEIQRDP"]["Xbeta"]:.2f}, sigma = {params["SEIQRDP"]["sigma"]*scales["SEIQRDP"]["Xsigma"]:.2f}, gamma = {params["SEIQRDP"]["gamma"]*scales["SEIQRDP"]["Xgamma"]:.2f}, lambda = {params["SEIQRDP"]["lambda"]:.2f}')

    ax4.relim()
    ax4.autoscale_view()
    fig.canvas.draw_idle()
def update_all(x):
    inis['E0'] = E0_slider.val
    inis['I0'] = I0_slider.val
    scales['time'] = int(time_slider.val)
    update_SEIR(x, update=False)
    update_SEIRP(x, update=False)
    update_SEIQR(x, update=False)
    update_SEIQRDP(x, update=False)

    update_params()

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
def create_real_plot(ax, name):
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

def create_SEIR_plot(ax, SEIR_test_result):
    lines = []
    colors = ['yellow', 'green', 'red', 'orange']
    for i in range(4):
        if 'SEIR'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIR_test_result[i], color=colors[i], label='SEIR'[i]))
    ax.title.set_text(f'beta = {params["SEIR"]["beta"]*scales["SEIR"]["Xbeta"]:.2f}, sigma = {params["SEIR"]["sigma"]*scales["SEIR"]["Xsigma"]:.2f}, gamma = {params["SEIR"]["gamma"]*scales["SEIR"]["Xgamma"]:.2f}')
    ax.set_ylabel('Proportion in\neach compartment')
    ax.legend()

    ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                r'$\frac{dR}{dt} = \gamma I$'
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')

    return lines
def create_SEIRP_plot(ax, SEIRP_test_result):
    lines = []
    colors = ['yellow', 'green', 'red', 'orange', 'purple']
    for i in range(5):
        if 'SEIRP'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIRP_test_result[i], color=colors[i], label='SEIRP'[i]))

    ax.title.set_text(f'alpha = {params["SEIRP"]["alpha"]*scales["SEIRP"]["Xalpha"]:.2f}, beta = {params["SEIRP"]["beta"]*scales["SEIRP"]["Xbeta"]:.2f}, sigma = {params["SEIRP"]["sigma"]*scales["SEIRP"]["Xsigma"]:.2f}, gamma = {params["SEIRP"]["gamma"]*scales["SEIRP"]["Xgamma"]:.2f}')
    ax.legend()
    ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                r'$\frac{dR}{dt} = \gamma I$' + '\n' + \
                r'$\frac{dP}{dt} = \alpha S$'
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines
def create_SEIQR_plot(ax, SEIQR_test_result):
    lines = []
    colors = ['yellow', 'green', 'red', 'blue', 'orange']
    for i in range(5):
        if 'SEIQR'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIQR_test_result[i], color=colors[i], label='SEIQR'[i]))

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion in\neach compartment')
    ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E - \delta I$' + '\n' + \
                r'$\frac{dQ}{dt} = \delta I - \gamma Q$' + '\n' + \
                r'$\frac{dR}{dt} = \gamma Q$'
    ax.title.set_text(f'beta = {params["SEIQR"]["beta"]*scales["SEIQR"]["Xbeta"]:.2f}, sigma = {params["SEIQR"]["sigma"]*scales["SEIQR"]["Xsigma"]:.2f}, gamma = {params["SEIQR"]["gamma"]*scales["SEIQR"]["Xgamma"]:.2f}, lambda = {params["SEIQR"]["lambda"]:.2f}')
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines
def create_SEIQRDP_plot(ax, SEIQRDP_test_result):
    lines = []
    colors = ['yellow', 'green', 'red', 'blue', 'orange', 'black', 'purple']
    for i in range(7):
        if 'SEIQRDP'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIQRDP_test_result[i], color=colors[i], label='SEIQRDP'[i]))
    ax.legend()
    ax.set_xlabel('Time')
    ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                r'$\frac{dQ}{dt} = \gamma I - \lambda(t) Q$ - ' + r'$\kappa(t) Q$' + '\n' + \
                r'$\frac{dR}{dt} = \lambda(t) Q$' + '\n' + \
                r'$\frac{dD}{dt} = \kappa(t) Q$' + '\n' + \
                r'$\frac{dP}{dt} = \alpha S$'
    ax4.title.set_text(f'alpha = {params["SEIQRDP"]["alpha"]*scales["SEIQRDP"]["Xalpha"]:.2f}, beta = {params["SEIQRDP"]["beta"]*scales["SEIQRDP"]["Xbeta"]:.2f}, sigma = {params["SEIQRDP"]["sigma"]*scales["SEIQRDP"]["Xsigma"]:.2f}, gamma = {params["SEIQRDP"]["gamma"]*scales["SEIQRDP"]["Xgamma"]:.2f}, lambda = {params["SEIQRDP"]["lambda"]:.2f}')

    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines
update_params()

#########################
##### CREATE FIGURE #####
#########################
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

SEIR_lines = create_SEIR_plot(ax1, SEIR_test(use_SEIQRDP))
SEIRP_lines = create_SEIRP_plot(ax2, SEIRP_test(use_SEIQRDP))
SEIQR_lines = create_SEIQR_plot(ax3, SEIQR_test(use_SEIQRDP))
SEIQRDP_lines = create_SEIQRDP_plot(ax4, SEIQRDP_test())

create_real_plot(ax1, 'SEIR')
create_real_plot(ax2, 'SEIRP')
create_real_plot(ax3, 'SEIQR')
create_real_plot(ax4, 'SEIQRDP')

##########################
##### CREATE SLIDERS #####
##########################
plt.subplots_adjust(top=.85, bottom=.18, hspace=0.7, wspace=0.7, left=.1, right=.8)
SEIR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.56, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)
SEIR_sigmaSlider = create_slider(ax=fig.add_axes([0.1, 0.53, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)
SEIR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.5, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)

SEIRP_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.56, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
SEIRP_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.53, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
SEIRP_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.5, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
SEIRP_alphaSlider = create_slider(ax=fig.add_axes([0.54, 0.47, 0.26, 0.03]), label='Xalpha', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)

SEIQR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.11, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIQR)
SEIQR_sigmaSlider = create_slider(ax=fig.add_axes([0.1, 0.08, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIQR)
SEIQR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.05, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIQR)
SEIQR_lambdaSlider = create_slider(ax=fig.add_axes([0.1, 0.02, 0.26, 0.03]), label='Xlambda', valmin=-2, valmax=2, valinit=0, callback=update_SEIQR)

SEIQRDP_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.11, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIQRDP)
SEIQRDP_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.08, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIQRDP)
SEIQRDP_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.05, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIQRDP)
SEIQRDP_alphaSlider = create_slider(ax=fig.add_axes([0.54, 0.02, 0.26, 0.03]), label='Xalpha', valmin=-2, valmax=2, valinit=0, callback=update_SEIQRDP)

E0_slider = create_slider(ax=fig.add_axes([0.1, 0.97, 0.7, 0.03]), label='E0', valmin=0.00000001, valmax=0.001, valinit=inis['E0'], callback=update_all)
I0_slider = create_slider(ax=fig.add_axes([0.1, 0.94, 0.7, 0.03]), label='I0', valmin=0.00000001, valmax=0.001, valinit=inis['I0'], callback=update_all)
time_slider = create_slider(ax=fig.add_axes([0.1, 0.91, 0.7, 0.03]), label='time', valmin=1, valmax=10, valinit=1, callback=update_all)

button = Button(fig.add_axes([0.9, 0.9, 0.085, 0.085]), 'Use SEIQRDP\nparams', color='lightgrey', hovercolor='0.9')
button.on_clicked(button_press)

fig.canvas.manager.full_screen_toggle()
plt.show()
