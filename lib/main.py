import h5py
import time
import logging
from numpy import savetxt, array
from datetime import datetime
from copy import deepcopy

from openquake.baselib.general import DictArray
from openquake.commonlib.readinput import get_oqparam, get_imts

from vengine.lib import imcm, parser, calc, plotutils  # IMPORT VPSHA MODULES


def run_job(job_ini, quantity = 'poe', calc_mode = 'full', nb_runs = 1, cm=imcm.BakerCornell2006()):
    """
    :param job_ini: str, path to Openquake configuration file (e.g. job.ini)
    :param quantity: str, quantity of interest: 'poe', 'are'
    :param calc_mode: str, calculation mode: Full hazard matrix computation ("full"),
                           or optimized search for vector samples matching POE/ARE ("optim")
    :param nb_runs: int, number of repetitive optimisation runs requested to
                         produce a set of solutions. If calc_mode=="full",
                         this option has no effect (=1).
    :param cm: imcm.IntensityMeasureCorrelationModel instance.
               Inter-intensity correlation model.
    """
    start_time = time.time()

    # Parse the seismic source model:
    oqparam = get_oqparam(job_ini)

    # Get the list of Tectonic Region Types :
    trt = oqparam._gsims_by_trt.keys()

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

    if calc_mode.lower().endswith("-marginals"):
        # First, run computation on 1-D PSHA for each period, and then run a "full" mode
        # N-D calculation and compute marginal 1-D hazard curves for comparison
        # Two modes available:
        # "plot-marginals" --> compute marginals, save to HDF5 and save plots
        # "calc-marginals", "full-with-marginals" --> compute marginals, save to HDF5, no plot.

        if calc_mode.lower().startswith("plot"):
            nsteps = 4
            build_plots = True
        elif calc_mode.lower().startswith("calc") or calc_mode.lower().startswith('full'):
            nsteps = 3
            build_plots = False
        else:
            raise ValueError(f'Unknown calculcation mode: "{calc_mode}"')
        
        # Suite of independent 1-D calculations:
        print(f'\n# STEP 1/{nsteps}: Compute unidimensional hazard curves for all periods')
        nd = len(oqparam.imtls.keys())
        ref1D = dict()
        for key in list(oqparam.imtls.keys()):
            print(f'### IMT: {str(key)} ###')
            # Initialize pqrm1D as a copy of oqparam:
            oqprm1D = deepcopy(oqparam)
            # Then, select only one IMT:
            oqprm1D.hazard_imtls = {str(key): oqparam.imtls[key].tolist()}
            c1D = calc.VectorValuedCalculator(oqprm1D, sites_col, cm)
            hc1D = c1D.hazard_matrix_calculation_parallel(quantity=quantity)
            ref1D[key] = {'imtls': deepcopy(oqprm1D.hazard_imtls),
                          'data': deepcopy(hc1D.hazard_matrix)}

        # N-D hazard matrix computation:
        print(f'\n# STEP 2/{nsteps} Compute N-dimensional hazard matrix')
        hc = c.hazard_matrix_calculation_parallel(quantity=quantity)

        # # Compute marginals:
        # marg_poe, marg_pdf = hc.hazard_matrix.compute_marginals()

        # Save all matrix/curves in HDF5:
        print(f'\n# STEP 3/{nsteps} Save results in HDF5 archive')
        results_file = '{}_with_marginals_'.format(quantity) + \
                       '{}.hdf5'.format(datetime.now().replace(microsecond=0).isoformat()).replace(':', '')
        with h5py.File(results_file, 'w') as h5f:
            dset = h5f.create_dataset('hazard_matrix', data=hc.hazard_matrix)
            for p in oqparam.imtls.keys():
                dset.attrs[p] = oqparam.imtls[p]
                dset1D = h5f.create_dataset(f'hazard_curve_{str(p)}', data=ref1D[key]['data'])
                dset1D.attrs[p] = array(ref1D[p]['imtls'][p])

        if build_plots:
            # Produce plots and save them:
            print(f'\n# STEP 4/{nsteps} Make plots and save to current directory')
            plotutils.plot_marginals(hc.hazard_matrix, oqparam.imtls, refcurves=ref1D, savedir='.')

    elif calc_mode.lower()=="full":
        # Next line distributes calculation over individual point-sources:
        #hc = c.hazard_matrix_calculation(quantity=quantity)

        # Next line distributes calculation over hazard-matrix cells:
        hc = c.hazard_matrix_calculation_parallel(quantity=quantity)

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
            results_file = '{}_{}'.format(quantity,trg) + \
                           '_{}.csv'.format(datetime.now().replace(microsecond=0).isoformat()).replace(':','')
            header_cols = [quantity.upper(), 'NITER', 'NFEV'] + [str(p) for p in c.periods]
            with open(results_file, 'w') as f:
                # Write header:
                f.write(','.join(header_cols)+'\n')
            optimization_method(trg, quantity=quantity, nsol=nb_runs, outputfile=results_file)

    else:
        raise ValueError('Unknown calculation mode "{}"'.format(calc_mode))

    logging.info('Results stored in file {}'.format(results_file))


