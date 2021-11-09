import numpy as np
import matplotlib.pyplot as plt
import metpy.plots.ctables as ctables

def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(figsize=(7, 6), nrows=nrows)
    fig.subplots_adjust(top=.93, bottom=0.01, left=0.32, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=ctables.registry.get_colortable(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()

cmaps = list(ctables.registry)
cmaps = [name for name in cmaps if name[-2:]!='_r']
nrows = len(cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

plot_color_gradients('MetPy', cmaps, nrows)
plt.show()