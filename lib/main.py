import h5py
import time
import logging
from numpy import savetxt, array
from datetime import datetime

from openquake.commonlib.readinput import get_oqparam, get_imts

from lib import imcm, parser, calc  # IMPORT VPSHA MODULES


def run_job(job_ini, quantity = 'poe', calc_mode = 'full', nb_runs = 1):
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
    logging.info(get_imts(oqparam))
    #    periods = oqparam.imtls.keys()

    # Initialize VPS§HA calculator:
    c = calc.VectorValuedCalculator(oqparam, sites_col, cm)

    # Count total number of point-sources:
    npts = c.count_pointsources()
    logging.info('\nNumber of point-sources: {}'.format(npts))

    if calc_mode.lower()=="full":
        # Next line distributes calculation over individual point-sources:
        #hc = c.hazard_matrix(quantity=quantity)

        # Next line distributes calculation over hazard-matrix cells:
        hc = c.hazard_matrix_parallel(quantity=quantity)

        # Save to HDF5 file:
        results_file = '{}_'.format(quantity) + \
                       '{}.hdf5'.format(datetime.now().replace(microsecond=0).isoformat()).replace(':','')
        with h5py.File(results_file, 'w') as h5f:
            dset = h5f.create_dataset('hazard_matrix', data=hc.hazard_matrix)
            for p in oqparam.imtls.keys():
                dset.attrs[str(p)] = oqparam.imtls[p]

    elif calc_mode.lower()=='optim':
        # Find POE by multi-dimensional optimization (e.g. Simplex method, Newton-Raphson method etc...)
        if quantity.lower()=='are':
            targets = calc.poe2are(oqparam.poes)
        elif quantity.lower()=='poe':
            targets = oqparam.poes
        else:
            raise ValueError('Unknown hazard curve quantity "{}"'.format(quantity))

        for trg in targets:
            logging.warning('Searching for pseudo-acceleration vector matching POE={}:'.format(trg))
            optimization_method = c.find_matching_poe # c.find_matching_poe_parallel_runs
            output = optimization_method(trg, quantity=quantity, nsol=nb_runs)
            results_file = '{}_{}'.format(quantity,trg) + \
                           '_{}.csv'.format(datetime.now().replace(microsecond=0).isoformat()).replace(':','')
            header_cols = [quantity.upper(), 'NITER', 'NFEV'] + [str(p) for p in c.periods]
            savetxt(results_file, output, fmt='%.6e', delimiter=',', header=','.join(header_cols))

    else:
        raise ValueError('Unknown calculation mode "{}"'.format(calc_mode))

    logging.info('Results stored in file {}'.format(results_file))


