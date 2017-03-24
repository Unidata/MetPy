.. _plots:

Plotting
========

Plots
-----

.. automodule:: metpy.plots

.. autosummary::
   :toctree: generated/

   SkewT
   SkewT.plot
   SkewT.plot_barbs
   SkewT.plot_dry_adiabats
   SkewT.plot_moist_adiabats
   SkewT.plot_mixing_lines
   Hodograph
   Hodograph.add_grid
   Hodograph.plot
   Hodograph.plot_colormapped
   StationPlot
   StationPlot.plot_symbol
   StationPlot.plot_parameter
   StationPlot.plot_text
   StationPlot.plot_barb
   StationPlotLayout
   StationPlotLayout.add_value
   StationPlotLayout.add_symbol
   StationPlotLayout.add_text
   StationPlotLayout.add_barb
   StationPlotLayout.names
   StationPlotLayout.plot


Colortables
-----------
:mod:`metpy.plots.ctables`

.. automodule:: metpy.plots.ctables

.. autosummary::
  :toctree: generated/

   read_colortable
   convert_gempak_table
   registry.scan_resource
   registry.scan_dir
   registry.add_colortable
   registry.get_with_steps
   registry.get_with_boundaries
   registry.get_colortable

.. plot::

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

   cmaps = list(ctables.registry.keys())
   nrows = len(cmaps)
   gradient = np.linspace(0, 1, 256)
   gradient = np.vstack((gradient, gradient))

   plot_color_gradients('MetPy', cmaps, nrows)
   plt.show()