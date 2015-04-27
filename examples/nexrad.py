import numpy as np
from numpy.ma import masked_array
import matplotlib.pyplot as plt
#from matplotlib.colors import Normalize
from scipy.constants import degree
from metpy.io.nexrad import Level3File
#from metpy.vis import ctables
#    name = 'testdata/KTLX20081110_220148_V03'
#    f = Level2File(name)
name = 'testdata/Level3_FFC_N0Q_20140407_1805.nids'
f = Level3File(name)
datadict = f.sym_block[0][0]

ref = np.array(datadict['data'])
ref = masked_array(ref, mask=(ref==32770))
az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
rng = np.arange(ref.shape[1] + 1)

#    if is_precip_mode(f.prod_desc.vcp):
#        norm = ctables.NWSRefClearAir
#    else:
#        norm = ctables.NWSRefPrecip

xlocs = rng * np.sin(az[:, np.newaxis] * degree)
ylocs = rng * np.cos(az[:, np.newaxis] * degree)
plt.pcolormesh(xlocs, ylocs, ref)#, cmap=ctables.NWSRef, norm=norm)
plt.colorbar()#ticks=np.linspace(norm.vmin, norm.vmax, 15))
plt.axis('equal')
plt.show()
