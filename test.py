
import imcm, mdhc, parser, calc  # IMPORT VPSHA MODULES
from openquake.commonlib.readinput import get_oqparam
import numpy as np
import h5py
from datetime import datetime
import time
import sys

# HELP:
# To laucnh the script use the following command:
#   python3 test.py "AreaSourceClassicalPSHA/job.ini"
#
# Below the following parameters can be changed by the user:
#  quantity = "poe" or "are"



start_time = time.time()
# Locate the job.ini file:
#ini_file = "AreaSourceClassicalPSHA/modified_job.ini"
#ini_file = "PointSourceClassicalPSHA/job.ini"
ini_file = sys.argv[1]

# Quantity of interest: 'poe', 'are'
quantity = 'poe'

# Parse the seismic source model:
oqparam = get_oqparam(ini_file)

# Get the list of Tectonic Region Types :
trt = oqparam._gsims_by_trt.keys()

# Set the Ground-motion correlation model:
# exists in Openquake :ground_motion_correlation_model
cm = imcm.BakerCornell2006()

# Manage the target sites specification, as site, site-collection or as a region:
sites_col = parser.parse_sites(oqparam)

# Decide whether (True) or not (False) to store the full ARE N-D matrix:
store_matrix = True

# Initialize multi-dimensional hazard curve:
periods = oqparam.imtls.keys()

# Initialize VPS§HA calculator:
c = calc.VectorValuedCalculator(oqparam, sites_col, cm)

# Count total number of point-sources:
print(f'\nNumber of point-sources: {c.count_pointsources()}')

if store_matrix:
        output = c.hazard_matrix(quantity=quantity, show_progress=True)
        # Save to HDF5 file:
        h5fname = f'{quantity}_{datetime.now().replace(microsecond=0).isoformat()}.hdf5'
        with h5py.File(h5fname, 'w') as h5f:
            h5f.create_dataset('output', data=output)
        print(f'\n>> Results stored in file {h5fname}')

else:
    # Find POE by multi-dimensional optimization (e.g. Simplex method, Newton-Raphson method etc...)
    accels = [0.1]*c.ndims
    poe = getattr(c,quantity)(accels)  # dimension: [ N sites x 1 ]
    # TODO: to be completed!

print(f'\nElapsed time: {time.time()-start_time} s.')


# TODO: Verifier coherence avec resultats OQ en 1-D: OK. Test passed! (April 23rd, 2020)
# TODO: Paralleliser le calcul de la poe
# TODO: option - Paralleliser pour un cluster ...
# TODO: Travailler sur l'approche par optimization
