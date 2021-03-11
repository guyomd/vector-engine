from matplotlib import pyplot as plt
from lib.plotutils import plot_matrix
import sys
import h5py
import numpy as np
from openquake.commonlib.readinput import get_oqparam

# HELP:
# This script is useful to plot "2-D Hazard curves" obtained by VPSHA.
# VPSHA output is saved into a .HDF5 file.
#
# To produce plot, use the following command:
#   python3 test_plot.py "AreaSourceClassicalPSHA/job.ini" "poe.hdf5"

ini_file = sys.argv[1]
h5fname = sys.argv[2]

oqparam = get_oqparam(ini_file)
periods = oqparam.imtls.keys()


accs = [oqparam.imtls[p] for p in periods]
per = list(periods)

with h5py.File(h5fname) as h5f:
    a = h5f.get('output')[()]
matrix = a[0,:,:]


plot_matrix(matrix, accs[1], accs[0], per[1], per[0], 'POE', ndigits_labels=3)
plt.show()

plt.Figure()
lev = np.logspace(-6, -1, 6)
x, y = np.meshgrid(accs[0], accs[1], indexing='ij')
cs = plt.contour(x, y, matrix, levels=lev)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(per[1])
ax.set_ylabel(per[0])
plt.clabel(cs, lev, fmt = '%.1e', fontsize=8)
plt.show()
