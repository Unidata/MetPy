import tempfile
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from metpy.plots.skewt import *  # noqa


# TODO: Need at some point to do image-based comparison, but that's a lot to
# bite off right now
class TestSkewT(object):
    def test_api(self):
        'Test the SkewT api'
        fig = Figure(figsize=(9, 9))
        skew = SkewT(fig)

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        p = np.linspace(1000, 100, 10)
        t = np.linspace(20, -20, 10)
        u = np.linspace(-10, 10, 10)
        skew.plot(p, t, 'r')
        skew.plot_barbs(p, u, u)

        # Add the relevant special lines
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        with tempfile.NamedTemporaryFile() as f:
            FigureCanvasAgg(fig).print_png(f.name)
