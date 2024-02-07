import numpy as np
import matplotlib.pyplot as plt
from models import SEIR_test, SEIRP_test, time
from matplotlib.widgets import Button, Slider
from models import scales

def plot_SEIR():
    SEIR_test_result = SEIR_test()

    fig, ax = plt.subplots()
    lines = []
    for result_array in SEIR_test_result:
        lines.append(ax.plot(time, result_array, color='blue'))

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')

    fig.subplots_adjust(left=0.25, bottom=0.25)

    betaSlider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),label='Xbeta',valmin=0.1,valmax=5,valinit=1)
    sigmaSlider = Slider(ax=fig.add_axes([0.25, 0.06, 0.65, 0.03]),label='Xsigma',valmin=0.1,valmax=5,valinit=1)
    gammaSlider = Slider(ax=fig.add_axes([0.25, 0.02, 0.65, 0.03]),label='Xgamma',valmin=0.1,valmax=5,valinit=1)
    def update(x):
        scales['SEIR']['Xbeta'] = betaSlider.val
        scales['SEIR']['Xsigma'] = sigmaSlider.val
        scales['SEIR']['Xgamma'] = gammaSlider.val

        SEIR_Update = SEIR_test()
        for i in range(4):
            lines[i][0].set_ydata(SEIR_Update[i])
        fig.canvas.draw_idle()
    betaSlider.on_changed(update)
    sigmaSlider.on_changed(update)
    gammaSlider.on_changed(update)

    plt.show()

def plot_SEIRP():
    SEIRP_test_result = SEIRP_test()

    fig, ax = plt.subplots()
    lines = []
    for result_array in SEIRP_test_result:
        lines.append(ax.plot(time, result_array, color='blue'))

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')

    fig.subplots_adjust(left=0.25, bottom=0.25)

    betaSlider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),label='Xbeta',valmin=0.1,valmax=5,valinit=1)
    sigmaSlider = Slider(ax=fig.add_axes([0.25, 0.06, 0.65, 0.03]),label='Xsigma',valmin=0.1,valmax=5,valinit=1)
    gammaSlider = Slider(ax=fig.add_axes([0.25, 0.02, 0.65, 0.03]),label='Xgamma',valmin=0.1,valmax=5,valinit=1)
    def update(x):
        scales['SEIRP']['Xbeta'] = betaSlider.val
        scales['SEIRP']['Xsigma'] = sigmaSlider.val
        scales['SEIRP']['Xgamma'] = gammaSlider.val

        SEIRP_Update = SEIRP_test()
        for i in range(5):
            lines[i][0].set_ydata(SEIRP_Update[i])
        fig.canvas.draw_idle()
    betaSlider.on_changed(update)
    sigmaSlider.on_changed(update)
    gammaSlider.on_changed(update)

    plt.show()

plot_SEIR()
#plot_SEIRP()
