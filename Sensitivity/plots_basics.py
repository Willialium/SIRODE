import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from models import SIS_test, SIR_test, SEI_test, SEIR_test, SEIRP_test, data
from matplotlib.widgets import Slider, Button
from models import scales, inis, params

EXCLUDE = 'S'
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

        temp_params['SIS']['beta'] *= scales['SIS']['Xbeta']

        temp_params['SIR']['beta'] *= scales['SIR']['Xbeta']
        temp_params['SIR']['gamma'] *= scales['SIR']['Xgamma']

        temp_params['SEIR']['beta'] *= scales['SEIR']['Xbeta']
        temp_params['SEIR']['sigma'] *= scales['SEIR']['Xsigma']
        temp_params['SEIR']['gamma'] *= scales['SEIR']['Xgamma']

        temp_params['SEIRP']['sigma'] *= scales['SEIRP']['Xsigma']
        temp_params['SEIRP']['beta'] *= scales['SEIRP']['Xbeta']
        temp_params['SEIRP']['gamma'] *= scales['SEIRP']['Xgamma']
        temp_params['SEIRP']['alpha'] *= scales['SEIRP']['Xalpha']

        f.write(json.dumps(temp_params, indent=3))

#############################
##### UPDATE FUNCTIONS  #####
#############################
def update_SIS(x, update=True):
    scales['SIS']['Xbeta'] = base**SIS_betaSlider.val
    if update: update_params()

    SIS_Update = SIS_test(use_SEIQRDP)
    model_buffer = 0

    for i in range(len('SI') - len(find_intersection('SI', EXCLUDE))):
        while 'SI'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SIS_lines[i][0].set_ydata(SIS_Update[i+model_buffer])
        SIS_lines[i][0].set_xdata(np.arange(0, len(SIS_Update[i+model_buffer])))
    ax1.relim()
    ax1.autoscale_view()
    ax1.title.set_text(f'beta = {params["SIS"]["beta"]*scales["SIS"]["Xbeta"]:.2f}')
    fig.canvas.draw_idle()
def update_SIR(x, update=True):
    scales['SIR']['Xbeta'] = base**SIR_betaSlider.val
    scales['SIR']['Xgamma'] = base**SIR_gammaSlider.val
    if update: update_params()

    SIR_Update = SIR_test(use_SEIQRDP)
    model_buffer = 0

    for i in range(len('SIR') - len(find_intersection('SIR', EXCLUDE))):
        while 'SIR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SIR_lines[i][0].set_ydata(SIR_Update[i+model_buffer])
        SIR_lines[i][0].set_xdata(np.arange(0, len(SIR_Update[i+model_buffer])))
    ax2.relim()
    ax2.autoscale_view()
    ax2.title.set_text(f'beta = {params["SIR"]["beta"]*scales["SIR"]["Xbeta"]:.2f}, gamma = {params["SIR"]["gamma"]*scales["SIR"]["Xgamma"]:.2f}')
    fig.canvas.draw_idle()
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
    ax3.relim()
    ax3.autoscale_view()
    ax3.title.set_text(f'beta = {params["SEIR"]["beta"]*scales["SEIR"]["Xbeta"]:.2f}, sigma = {params["SEIR"]["sigma"]*scales["SEIR"]["Xsigma"]:.2f}, gamma = {params["SEIR"]["gamma"]*scales["SEIR"]["Xgamma"]:.2f}')
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

    ax4.relim()
    ax4.autoscale_view()
    ax4.title.set_text(f'alpha = {params["SEIRP"]["alpha"]*scales["SEIRP"]["Xalpha"]:.2f}, beta = {params["SEIRP"]["beta"]*scales["SEIRP"]["Xbeta"]:.2f}, sigma = {params["SEIRP"]["sigma"]*scales["SEIRP"]["Xsigma"]:.2f}, gamma = {params["SEIRP"]["gamma"]*scales["SEIRP"]["Xgamma"]:.2f}')
    fig.canvas.draw_idle()
def update_SEI(x, update=True):
    scales['SEI']['Xbeta'] = base**SEI_betaSlider.val
    scales['SEI']['Xsigma'] = base**SEI_sigmaSlider.val
    if update: update_params()

    SEI_Update = SEI_test(use_SEIQRDP)
    model_buffer = 0
    for i in range(len('SEI') - len(find_intersection('SEI', EXCLUDE))):
        while 'SEI'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEI_lines[i][0].set_ydata(SEI_Update[i+model_buffer])
        SEI_lines[i][0].set_xdata(np.arange(0, len(SEI_Update[i+model_buffer])))

    ax4.relim()
    ax4.autoscale_view()
    ax4.title.set_text(f'beta = {params["SEI"]["beta"]*scales["SEI"]["Xbeta"]:.2f}, gamma = {params["SEI"]["sigma"]*scales["SEI"]["Xsigma"]:.2f}')
    fig.canvas.draw_idle()
def update_all(x):
    inis['E0'] = E0_slider.val
    inis['I0'] = I0_slider.val
    scales['time'] = int(time_slider.val)
    update_SIS(x, update=False)
    update_SIR(x, update=False)
    update_SEIR(x, update=False)
    #update_SEIRP(x, update=False)
    update_SEI(x, update=False)

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


def create_SIS_plot(ax, SIS_test_result):
    lines = []
    colors = ['yellow', 'red']
    for i in range(2):
        if 'SI'[i] not in EXCLUDE:
            lines.append(ax.plot(SIS_test_result[i], color=colors[i], label='SI'[i]))
    ax.title.set_text(f'beta = {params["SIS"]["beta"]*scales["SIS"]["Xbeta"]:.2f}')
    ax.set_ylabel('Proportion in\neach compartment')
    ax.legend()

    ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                r'$\frac{dI}{dt} = \beta SI$' + '\n'

    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')

    return lines
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
def create_SEI_plot(ax, SEI_test_result):
    lines = []
    colors = ['yellow', 'green', 'red']
    for i in range(3):
        if 'SEI'[i] not in EXCLUDE:
            lines.append(ax.plot(SEI_test_result[i], color=colors[i], label='SEI'[i]))

    ax.title.set_text(f'beta = {params["SEI"]["beta"]*scales["SEI"]["Xbeta"]:.2f}, sigma = {params["SEI"]["sigma"]*scales["SEI"]["Xsigma"]:.2f}')
    ax.legend()
    ode_latex = r'$\frac{dS}{dt} = -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E$'
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines
update_params()

#########################
##### CREATE FIGURE #####
#########################
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

SIS_lines = create_SIS_plot(ax1, SIS_test(use_SEIQRDP))
SIR_lines = create_SIR_plot(ax2, SIR_test(use_SEIQRDP))
SEIR_lines = create_SEIR_plot(ax3, SEIR_test(use_SEIQRDP))
#SEIRP_lines = create_SEIRP_plot(ax4, SEIRP_test(use_SEIQRDP))
SEI_lines = create_SEI_plot(ax4, SEI_test(use_SEIQRDP))

create_real_plot(ax1, 'SIS')
create_real_plot(ax2, 'SIR')
create_real_plot(ax3, 'SEIR')
#create_real_plot(ax4, 'SEIRP')
create_real_plot(ax4, 'SEI')


##########################
##### CREATE SLIDERS #####
##########################
plt.subplots_adjust(top=.85, bottom=.18, hspace=0.7, wspace=0.7, left=.1, right=.8)

SIS_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.56, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SIS)

SIR_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.56, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SIR)
SIR_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.53, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SIR)

SEIR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.11, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)
SEIR_sigmaSlider = create_slider(ax=fig.add_axes([0.1, 0.08, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)
SEIR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.05, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIR)

#SEIRP_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.11, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
#SEIRP_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.08, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
#SEIRP_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.05, 0.26, 0.03]), label='Xgamma', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)
#SEIRP_alphaSlider = create_slider(ax=fig.add_axes([0.54, 0.02, 0.26, 0.03]), label='Xalpha', valmin=-2, valmax=2, valinit=0, callback=update_SEIRP)

SEI_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.11, 0.26, 0.03]), label='Xbeta', valmin=-2, valmax=2, valinit=0, callback=update_SEI)
SEI_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.08, 0.26, 0.03]), label='Xsigma', valmin=-2, valmax=2, valinit=0, callback=update_SEI)

E0_slider = create_slider(ax=fig.add_axes([0.1, 0.97, 0.7, 0.03]), label='E0', valmin=0.00000001, valmax=0.001, valinit=inis['E0'], callback=update_all)
I0_slider = create_slider(ax=fig.add_axes([0.1, 0.94, 0.7, 0.03]), label='I0', valmin=0.00000001, valmax=0.001, valinit=inis['I0'], callback=update_all)
time_slider = create_slider(ax=fig.add_axes([0.1, 0.91, 0.7, 0.03]), label='time', valmin=1, valmax=10, valinit=1, callback=update_all)

button = Button(fig.add_axes([0.9, 0.9, 0.085, 0.085]), 'Use SEIQRDP\nparams', color='lightgrey', hovercolor='0.9')
button.on_clicked(button_press)

fig.canvas.manager.full_screen_toggle()
plt.show()
