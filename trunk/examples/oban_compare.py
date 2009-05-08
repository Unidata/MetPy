import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase

from scipy.interpolate import Rbf
from matplotlib.mlab import griddata
from metpy.tools.oban import (grid_data, cressman_weights, barnes_weights,
    gaussian_filter)

def rms(data):
    return np.sqrt((data**2).mean())

# Observations
rs = np.random.RandomState(1353235)
noise = rs.randn(100)*.01

x = rs.rand(100)*4.0-2.0
y = rs.rand(100)*4.0-2.0
z = x*np.exp(-x**2-y**2) + noise

# Set up the grid
ti = np.linspace(-2.0, 2.0, 100)
XI, YI = np.meshgrid(ti, ti)

# True analytic field
truth = XI*np.exp(-XI**2-YI**2)

# use RBF - multiquadric
epsm = 0.15
rbfm = Rbf(x, y, z, function='multiquadric', epsilon=epsm, smooth=0.05)
rbfm_interp = rbfm(XI, YI)
rbfm_rms = rms(rbfm_interp - truth)

# use RBF - gaussian
epsg = .4
rbfg = Rbf(x, y, z, function='gaussian', epsilon=epsg, smooth=0.1)
rbfg_interp = rbfg(XI, YI)
rbfg_rms = rms(rbfg_interp - truth)

# Natural neighbor
nat_interp = griddata(x, y, z, XI, YI)
nat_filtered = gaussian_filter(XI, YI, nat_interp, 0.1, 0.1)
nat_rms = rms(nat_interp - truth)

# Cressman
Rc = 0.5
cress_interp = grid_data(z, XI, YI, x, y, cressman_weights, Rc)
cress_rms = rms(cress_interp - truth)

# Barnes
kstar = 0.1
barnes_interp = grid_data(z, XI, YI, x, y, barnes_weights, (0.5, kstar))
barnes_rms = rms(barnes_interp - truth)

# plot the results
val_norm = plt.normalize(-0.45, 0.45)
diff_norm = plt.normalize(-1.0, 1.0)
cmap = plt.get_cmap('jet')

fig = plt.figure()
fig.suptitle('Interpolation Results', fontsize=14)
fig.canvas.manager.set_window_title('Actual Fields')

ax = plt.subplot(2, 3, 1)
plt.pcolor(XI, YI, truth, cmap=cmap, norm=val_norm)
plt.scatter(x, y, 80, z, cmap=cmap, norm=val_norm)
plt.title('Truth')

ax2 = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, rbfm_interp, cmap=cmap, norm=val_norm)
plt.title('RBF - Multiquadrics $(\epsilon=%.1f)$' % epsm)

ax3 = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, rbfg_interp, cmap=cmap, norm=diff_norm)
plt.title('RBF - Gaussian $(\epsilon=%.2f)$' % epsg)

ax4 = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, nat_interp, cmap=cmap, norm=val_norm)
plt.title('Natural Neighbor')

ax5 = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, cress_interp, cmap=cmap, norm=val_norm)
plt.title('Cressman (R=%.2f)' % Rc)

ax6 = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, barnes_interp, cmap=cmap, norm=val_norm)
plt.title('Barnes $(\kappa=%.2f)$' % kstar)

plt.xlim(-2, 2)
plt.ylim(-2, 2)

#Make a big colorbar
fig.subplots_adjust(bottom=0.15)
lpos = [0.04, 0.04, 0.94, 0.04]
cax = fig.add_axes(lpos)
cbar = ColorbarBase(ax=cax, norm=val_norm, cmap=cmap, orientation='horizontal')

fig = plt.figure()
fig.suptitle('Interpolation Differences', fontsize=14)
fig.canvas.manager.set_window_title('Difference Fields')

ax = plt.subplot(2, 3, 1)
plt.pcolor(XI, YI, truth, cmap=cmap, norm=val_norm)
plt.scatter(x, y, 80, z, cmap=cmap, norm=val_norm)
plt.title('Truth')

ax2 = plt.subplot(2, 3, 2, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, (rbfm_interp - truth) / truth, cmap=cmap, norm=diff_norm)
plt.title('RBF - Multiquadrics $(\epsilon=%.2f)$: RMS %.3f' % (epsm, rbfm_rms))

ax3 = plt.subplot(2, 3, 3, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, (rbfg_interp - truth) / truth, cmap=cmap, norm=diff_norm)
plt.title('RBF - Gaussian $(\epsilon=%.2f)$: RMS %.3f' % (epsg, rbfg_rms))

ax4 = plt.subplot(2, 3, 4, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, (nat_interp - truth) / truth, cmap=cmap, norm=diff_norm)
plt.title('Natural Neighbor: RMS %.3f' % nat_rms)

ax5 = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, (cress_interp - truth) / truth, cmap=cmap, norm=diff_norm)
plt.title('Cressman (R=%.2f): RMS %.3f' % (Rc, cress_rms))

ax6 = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
plt.pcolor(XI, YI, (barnes_interp - truth) / truth, cmap=cmap, norm=diff_norm)
plt.title('Barnes $(\kappa=%.2f)$: RMS %.3f' % (kstar, barnes_rms))

plt.xlim(-2, 2)
plt.ylim(-2, 2)

#Make a big colorbar
fig.subplots_adjust(bottom=0.15)
lpos = [0.04, 0.04, 0.94, 0.04]
cax = fig.add_axes(lpos)
cbar = ColorbarBase(ax=cax, norm=diff_norm, cmap=cmap, orientation='horizontal')

plt.show()
