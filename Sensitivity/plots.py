import numpy as np
import matplotlib.pyplot as plt
from models import SEIR_test, SEIRP_test, SEIQR_test, SEIQDRP_test
from matplotlib.widgets import Slider
from models import scales, inis

EXCLUDE = 'SP'

# Function to create sliders
def create_slider(ax, label, valmin, valmax, valinit, callback):
    slider = Slider(ax=ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
    slider.on_changed(callback)
    return slider

#############################
##### UPDATE FUNCTIONS  #####
#############################

def update_SEIR(x):
    scales['SEIR']['Xbeta'] = SEIR_betaSlider.val
    scales['SEIR']['Xsigma'] = SEIR_sigmaSlider.val
    scales['SEIR']['Xgamma'] = SEIR_gammaSlider.val

    SEIR_Update = SEIR_test()
    model_buffer = 0
    for i in range(len('SEIR') - len(EXCLUDE)):
        if 'SEIR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIR_lines[i][0].set_ydata(SEIR_Update[i+model_buffer])
        SEIR_lines[i][0].set_xdata(np.arange(0, len(SEIR_Update[i+model_buffer])))
    ax1.relim()
    ax1.autoscale_view()
    fig.canvas.draw_idle()
def update_SEIRP(x):
    scales['SEIRP']['Xbeta'] = SEIRP_betaSlider.val
    scales['SEIRP']['Xsigma'] = SEIRP_sigmaSlider.val
    scales['SEIRP']['Xgamma'] = SEIRP_gammaSlider.val
    scales['SEIRP']['Xalpha'] = SEIRP_alphaSlider.val

    SEIRP_Update = SEIRP_test()
    model_buffer = 0
    for i in range(len('SEIRP') - len(EXCLUDE)):
        if 'SEIRP'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIRP_lines[i][0].set_ydata(SEIRP_Update[i+model_buffer])
        SEIRP_lines[i][0].set_xdata(np.arange(0, len(SEIRP_Update[i+model_buffer])))

    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw_idle()
def update_SEIQR(x):
    scales['SEIQR']['Xbeta'] = SEIQR_betaSlider.val
    scales['SEIQR']['Xsigma'] = SEIQR_sigmaSlider.val
    scales['SEIQR']['Xgamma'] = SEIQR_gammaSlider.val
    scales['SEIQR']['Xlaambda'] = SEIQR_lambdaSlider.val

    SEIQR_Update = SEIQR_test()
    model_buffer = 0
    for i in range(len('SEIQR') - len(EXCLUDE)):
        if 'SEIQR'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIQR_lines[i][0].set_ydata(SEIQR_Update[i+model_buffer])
        SEIQR_lines[i][0].set_xdata(np.arange(0, len(SEIQR_Update[i+model_buffer])))

    ax3.relim()
    ax3.autoscale_view()
    fig.canvas.draw_idle()
def update_SEIQDRP(x):
    scales['SEIQDRP']['Xbeta'] = SEIQDRP_betaSlider.val
    scales['SEIQDRP']['Xsigma'] = SEIQDRP_sigmaSlider.val
    scales['SEIQDRP']['Xgamma'] = SEIQDRP_gammaSlider.val
    scales['SEIQDRP']['Xalpha'] = SEIQDRP_alphaSlider.val

    SEIQDRP_Update = SEIQDRP_test()
    model_buffer = 0
    for i in range(len('SEIQDRP') - len(EXCLUDE)):
        if 'SEIQDRP'[i+model_buffer] in EXCLUDE:
            model_buffer += 1
        SEIQDRP_lines[i][0].set_ydata(SEIQDRP_Update[i+model_buffer])
        SEIQDRP_lines[i][0].set_xdata(np.arange(0, len(SEIQDRP_Update[i+model_buffer])))

    ax4.relim()
    ax4.autoscale_view()
    fig.canvas.draw_idle()
def update_all(x):
    inis['E0'] = E0_slider.val
    inis['I0'] = I0_slider.val
    inis['R0'] = R0_slider.val
    scales['time'] = int(time_slider.val)
    update_SEIR(x)
    update_SEIRP(x)
    update_SEIQR(x)
    update_SEIQDRP(x)

########################
##### CREATE PLOTS #####
########################

def create_SEIR_plot(ax, SEIR_test_result):
    lines = []
    colors = ['blue', 'green', 'red', 'black']
    for i in range(4):
        if 'SEIR'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIR_test_result[i], color=colors[i], label='SEIR'[i]))

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
    colors = ['blue', 'green', 'red', 'black', 'orange']
    for i in range(5):
        if 'SEIRP'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIRP_test_result[i], color=colors[i], label='SEIRP'[i]))

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
    colors = ['blue', 'green', 'red', 'black', 'orange']
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
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines
def create_SEIQDRP_plot(ax, SEIQDRP_test_result):
    lines = []
    colors = ['blue', 'green', 'red', 'black', 'orange', 'purple', 'yellow']
    for i in range(7):
        if 'SEIQDRP'[i] not in EXCLUDE:
            lines.append(ax.plot(SEIQDRP_test_result[i], color=colors[i], label='SEIQDRP'[i]))
    ax.legend()
    ax.set_xlabel('Time')
    ode_latex = r'$\frac{dS}{dt} = -\alpha S -\beta SI$' + '\n' + \
                r'$\frac{dE}{dt} = \beta SI - \sigma E$' + '\n' + \
                r'$\frac{dI}{dt} = \sigma E - \gamma I$' + '\n' + \
                r'$\frac{dQ}{dt} = \gamma I - \lambda(t) Q$ - ' + r'$\kappa(t) Q$' + '\n' + \
                r'$\frac{dR}{dt} = \lambda(t) Q$' + '\n' + \
                r'$\frac{dD}{dt} = \kappa(t) Q$' + '\n' + \
                r'$\frac{dP}{dt} = \alpha S$'
    ax.annotate(ode_latex, xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')
    return lines

#########################
##### CREATE FIGURE #####
#########################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

SEIR_lines = create_SEIR_plot(ax1, SEIR_test())
SEIRP_lines = create_SEIRP_plot(ax2, SEIRP_test())
SEIQR_lines = create_SEIQR_plot(ax3, SEIQR_test())
SEIQDRP_lines = create_SEIQDRP_plot(ax4, SEIQDRP_test())

##########################
##### CREATE SLIDERS #####
##########################

plt.subplots_adjust(top=.88, bottom=.25, hspace=0.65, wspace=0.7, left=.1, right=.8)
SEIR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.6, 0.26, 0.03]), label='Xbeta', valmin=0.1, valmax=5, valinit=1, callback=update_SEIR)
SEIR_sigmaSlider = create_slider(ax=fig.add_axes([0.1, 0.57, 0.26, 0.03]), label='Xsigma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIR)
SEIR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.54, 0.26, 0.03]), label='Xgamma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIR)

SEIRP_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.6, 0.26, 0.03]), label='Xbeta', valmin=0.1, valmax=5, valinit=1, callback=update_SEIRP)
SEIRP_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.57, 0.26, 0.03]), label='Xsigma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIRP)
SEIRP_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.54, 0.26, 0.03]), label='Xgamma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIRP)
SEIRP_alphaSlider = create_slider(ax=fig.add_axes([0.54, 0.51, 0.26, 0.03]), label='Xalpha', valmin=0.1, valmax=5, valinit=1, callback=update_SEIRP)

SEIQR_betaSlider = create_slider(ax=fig.add_axes([0.1, 0.17, 0.26, 0.03]), label='Xbeta', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQR)
SEIQR_sigmaSlider = create_slider(ax=fig.add_axes([0.1, 0.14, 0.26, 0.03]), label='Xsigma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQR)
SEIQR_gammaSlider = create_slider(ax=fig.add_axes([0.1, 0.11, 0.26, 0.03]), label='Xgamma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQR)
SEIQR_lambdaSlider = create_slider(ax=fig.add_axes([0.1, 0.08, 0.26, 0.03]), label='Xlambda', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQR)

SEIQDRP_betaSlider = create_slider(ax=fig.add_axes([0.54, 0.17, 0.26, 0.03]), label='Xbeta', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQDRP)
SEIQDRP_sigmaSlider = create_slider(ax=fig.add_axes([0.54, 0.14, 0.26, 0.03]), label='Xsigma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQDRP)
SEIQDRP_gammaSlider = create_slider(ax=fig.add_axes([0.54, 0.11, 0.26, 0.03]), label='Xgamma', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQDRP)
SEIQDRP_alphaSlider = create_slider(ax=fig.add_axes([0.54, 0.08, 0.26, 0.03]), label='Xalpha', valmin=0.1, valmax=5, valinit=1, callback=update_SEIQDRP)

E0_slider = create_slider(ax=fig.add_axes([0.1, 0.97, 0.7, 0.03]), label='E0', valmin=0.00000001, valmax=0.001, valinit=inis['E0'], callback=update_all)
I0_slider = create_slider(ax=fig.add_axes([0.1, 0.94, 0.7, 0.03]), label='I0', valmin=0.00000001, valmax=0.001, valinit=inis['I0'], callback=update_all)
R0_slider = create_slider(ax=fig.add_axes([0.1, 0.91, 0.7, 0.03]), label='R0', valmin=0.000000001, valmax=0.001, valinit=inis['R0'], callback=update_all)
time_slider = create_slider(ax=fig.add_axes([0.1, 0.88, 0.7, 0.03]), label='time', valmin=1, valmax=10, valinit=1, callback=update_all)


plt.show()
