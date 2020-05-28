
from lib import imcm, mdhc, parser, calc  # IMPORT VPSHA MODULES
from openquake.commonlib.readinput import get_oqparam, get_imts
from openquake.baselib.parallel import Starmap
from numpy import savetxt
import h5py
from datetime import datetime
import time
import sys

# HELP:
# To laucnh the script use the following command:
#   python3 vpsha.py "AreaSourceClassicalPSHA/job.ini"
#
# Below the following parameters can be changed by the user:
#  quantity = "poe" or "are"

def main(job_ini, quantity = 'poe', calc_mode = 'full'):
    """
    :param job_ini: str, path to Openquake configuration file (e.g. job.ini)
    :param quantity: str, quantity of interest: 'poe', 'are'
    :param calc_mode: str, calculation mode: Full hazard matrix computation ("full"),
                           or optimized search for vector samples matching POE/ARE ("optim")
    """
    start_time = time.time()

    # Parse the seismic source model:
    oqparam = get_oqparam(job_ini)

    # Get the list of Tectonic Region Types :
    trt = oqparam._gsims_by_trt.keys()

    # Set the Ground-motion correlation model:
    # exists in Openquake :ground_motion_correlation_model
    cm = imcm.BakerCornell2006()

    # Manage the target sites specification, as site, site-collection or as a region:
    sites_col = parser.parse_sites(oqparam)

    # Initialize multi-dimensional hazard curve:
    print(get_imts(oqparam))
    #    periods = oqparam.imtls.keys()


    # Initialize VPS§HA calculator:
    c = calc.VectorValuedCalculator(oqparam, sites_col, cm)

    # Count total number of point-sources:
    npts = c.count_pointsources()
    print(f'\nNumber of point-sources: {npts}')

    if calc_mode.lower()=="full":
        # Next line distributes calculation over hazard-matrix cells:
        #hc = c.hazard_matrix(quantity=quantity, show_progress=True)

        # Next line distributes calculation over individual point-sources:
        hc = c.hazard_matrix_parallel(quantity=quantity, show_progress=True)

        # Save to HDF5 file:
        results_file = f'{quantity}_' \
                       f'{datetime.now().replace(microsecond=0).isoformat()}.hdf5'.replace(':','')
        with h5py.File(results_file, 'w') as h5f:
            h5f.create_dataset('output', data=hc.hazard_matrix)

    elif calc_mode.lower()=='optim':
        # Find POE by multi-dimensional optimization (e.g. Simplex method, Newton-Raphson method etc...)
        if quantity.lower()=='are':
            targets = calc.poe2are(oqparam.poes)
        elif quantity.lower()=='poe':
            targets = oqparam.poes
        else:
            raise ValueError(f'Unknown hazard curve quantity "{quantity}"')

        n_sol = 10  # Number of vector-sample matching target
        for trg in targets:
            print(f'Searching for pseudo-acceleration vector matching POE={trg}:')
            output = c.find_matching_vector_sample(trg,
                                                   quantity=quantity,
                                                   tol=0.001,
                                                   n_real=n_sol)
            results_file = f'{quantity}_{trg}' \
                           f'_{datetime.now().replace(microsecond=0).isoformat()}.csv'.replace(':','')
            header_cols = [quantity.upper()] + [str(p) for p in c.periods]
            savetxt(results_file, output, fmt='%.6e', delimiter=',', header=','.join(header_cols))
            print(output)

    else:
        raise ValueError(f'Unknown calculation mode "{calc_mode}"')

    print(f'\n>> Results stored in file {results_file}')
    print(f'\nElapsed time: {time.time()-start_time:.3f} s.')


if __name__ == "__main__":
    Starmap.init()
    try:
        main(sys.argv[1], quantity='poe', calc_mode='optim')
    finally:
        Starmap.shutdown()

