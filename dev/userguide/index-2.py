import matplotlib.pyplot as plt
import numpy as np
from metpy.cbook import get_test_data
from metpy.io import Level3File
from metpy.plots import colortables

name = get_test_data(f'nids/KOUN_SDUS54_N0QTLX_201305202016', as_file_obj=False)
f = Level3File(name)
datadict = f.sym_block[0][0]
data = f.map_data(datadict['data'])

az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
rng = np.linspace(0, f.max_range, data.shape[-1] + 1)
xlocs = rng * np.sin(np.deg2rad(az[:, np.newaxis]))
ylocs = rng * np.cos(np.deg2rad(az[:, np.newaxis]))

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(4, 4), dpi=150)
norm, cmap = colortables.get_with_steps('NWSStormClearReflectivity', -20, 0.5)
ax.pcolormesh(xlocs, ylocs, data, norm=norm, cmap=cmap)
ax.set_aspect('equal', 'datalim')
ax.set_xlim(-40, 20)
ax.set_ylim(-30, 30)
ax.set_title('KTLX Reflectivity at 2013/05/20 2016Z')

plt.show()