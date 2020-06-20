import logging
from lib import parser, mdhc
from plotutils import ProgressBar
import numpy as np
from scipy.stats import mvn
from scipy.optimize import minimize, Bounds

from openquake.hazardlib.contexts import ContextMaker, RuptureContext

from openquake.commonlib.readinput import get_gsim_lt, get_source_model_lt, get_imts
from openquake.hazardlib import const

from openquake.baselib.parallel import Starmap
from openquake.baselib.datastore import hdf5new


def are2poe(are: np.ndarray, investigation_time=1.0):
    """
      Convert annual rates of exceedance (ARE) to probability of exceedance (POE),
      assuming a Poisson model and an investigationTime (or period) expressed in years
    """
    return 1 - np.exp(-are * investigation_time)


def poe2are(poe: np.ndarray, investigation_time=1.0):
    """
      Convert Probabilities Of ground-shaking Exceedance (POEs) to Annual
      Rates of Exceedance (ARE), assuming a Poisson model hypothesis
    """
    return -np.log(1 - poe) / float(investigation_time)


class VectorValuedCalculator():

    def __init__(self, oqparam, sites_col, correlation_model):
        self.oqparam = oqparam
        self.ssm_lt = get_source_model_lt(oqparam) # Read the SSC logic tree
        self.hc = mdhc.MultiDimensionalHazardCurve(oqparam.imtls,
                                                   sites_col, correlation_model,
                                                   oqparam.maximum_distance)
        self.ndims = len(oqparam.imtls.keys())
        #self.periods = oqparam.imtls.keys()
        self.periods = get_imts(oqparam)
        self.sites = sites_col
        self.cm = correlation_model
        self.truncation_level = oqparam.truncation_level


    def count_pointsources(self):
        """
        Returns the total number of point-sources involved in the current calculation
        """
        n = 0
        for rlz in self.ssm_lt:
            srcs = parser.get_sources_from_rlz(rlz, self.oqparam, self.ssm_lt)
            for src in srcs:
                for _ in src:
                    n += 1
        return n


    def gm_poe(self, gsim, dist_ctx, rup_ctx, site_ctx, lnSA):
        """
        Returns the multivariate probability to exceed a specified set of
        ground motion acceleration levels
        """
        lower = np.array(lnSA) # Natural logarithm of acceleration values
        abseps = 0.0001  # Documentation: Optimal value is 1E-6
        maxpts = len(lower)*10  # Documentation: Optimal value is len(lower)*1000
        nsites = len(site_ctx)
        lnAVG = np.zeros((nsites,self.ndims))
        lnSTD = np.zeros(lnAVG.shape)
        for i in range(self.ndims):
            # gsim().get_mean_and_stddevs returns ground-motion as ln(PSA) in units of g:
            means, stddevs = gsim.get_mean_and_stddevs(site_ctx,
                                                       rup_ctx,
                                                       dist_ctx,
                                                       self.periods[i],
                                                       [const.StdDev.TOTAL])
            lnAVG[:,i] = np.squeeze(means)
            lnSTD[:,i] = np.squeeze(stddevs[0])
        prob = np.zeros((nsites,))
        for j in range(nsites):
            # Build covariance matrix:
            D = np.diag(lnSTD[j,:])
            cov = D @ self.hc.corr @ D
            lower_trunc = lnAVG[j,:]-self.truncation_level*np.sqrt(np.diag(cov))
            upper_trunc = lnAVG[j,:]+self.truncation_level*np.sqrt(np.diag(cov))
            if np.any(lower >= upper_trunc):
                # Requested value is above the 3-sigma truncature for at least one spectral period:
                prob[j] = 0
            else:
                trunc_norm, _ = mvn.mvnun(lower_trunc,
                                   upper_trunc,
                                   lnAVG[j,:],
                                   cov,
                                   abseps=abseps,
                                   maxpts=maxpts)

                prob[j], error = mvn.mvnun(lower,
                                  upper_trunc,
                                  lnAVG[j,:],
                                  cov,
                                  abseps=abseps,
                                  maxpts=maxpts)
                # Normalize poe over the truncation interval [-n*sigma, n*sigma]
                prob[j] /= trunc_norm
        return prob


    def pt_src_are(self, pt_src, gsim, weight, lnSA, monitor):
        """
        Returns the vector-valued Annual Rate of Exceedance for one single point-source

        :param pt_src: single instance of class "openquake.hazardlib.source.area.PointSource"
        :param gsim: tuple, containing (only one?) instance of Openquake GSIM class
        :param: weight, weight to be multiplied to ARE estimate
        :param lnSA: list, natural logarithm of acceleration values for each spectral period.
            Note : Values should be ordered in the same order than self.periods
        """
        annual_rate = 0

        # Loop over ruptures:
        # i.e. one rupture for each combination of (mag, nodal plane, hypocentral depth):
        for r in pt_src.iter_ruptures():
        # NOTE: IF ACCOUNTING FOR "pointsource_distance" IN THE INI FILE, ONE SHOULD USE THE
        # "point_ruptures()" METHOD BELOW:
        # Loop over ruptures, one rupture for each magnitude ( neglect floating and combination on
        # nodal plane and hypocentral depth):
        ## for r in pt_src.point_ruptures():
            # Note: Seismicity rate evenly distributed over all point sources
            #       Seismicity rate also accounts for FMD (i.e. decreasing for
            #         increasing magnitude value)

            # Filter the site collection with respect to the rupture and prepare context objects:
            context_maker = ContextMaker(r.tectonic_region_type, gsim)
            site_ctx, dist_ctx = context_maker.make_contexts(self.sites, r)
            rup_ctx = RuptureContext()
            rup_ctx.mag = r.mag
            rup_ctx.rake = r.rake
            assert len(gsim)==1

            annual_rate += r.occurrence_rate * weight * self.gm_poe(gsim[0],
                                                                    dist_ctx,
                                                                    rup_ctx,
                                                                    site_ctx,
                                                                    lnSA)
        return annual_rate


    def are(self, lnSA):
        """
        Returns the vector-valued annual rate of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        are = 0
        for rlz in self.ssm_lt:  # Loop over realizations
            _, weight = parser.get_value_and_weight_from_rlz(rlz)
            srcs = parser.get_sources_from_rlz(rlz, self.oqparam, self.ssm_lt)

            for src in srcs:  # Loop over seismic sources (area, fault, etc...)

                for pt in src:  # Loop over point-sources

                    gsim_lt = get_gsim_lt(self.oqparam, trts=[src.tectonic_region_type])
                    for gsim_rlz in gsim_lt:  # Loop over GSIM Logic_tree
                        gsim_model, gsim_weight = parser.get_value_and_weight_from_gsim_rlz(
                            gsim_rlz)
                        pt_weight = weight * gsim_weight
                        are += self.pt_src_are(pt, gsim_model, pt_weight, lnSA, None)
        return are

    def are_parallel(self, lnSA):
        """
        Returns the vector-valued annual rate of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        args_list = list()
        #with hdf5new() as hdf5:
            #smap = Starmap(self.pt_src_are.__func__, h5=hdf5)
        for rlz in self.ssm_lt:  # Loop over realizations
            _, weight = parser.get_value_and_weight_from_rlz(rlz)   
            srcs = parser.get_sources_from_rlz(rlz, self.oqparam, self.ssm_lt)

            for src in srcs:  # Loop over seismic sources (area, fault, etc...)

                for pt in src:  # Loop over point-sources

                    gsim_lt = get_gsim_lt(self.oqparam, trts=[src.tectonic_region_type])
                    for gsim_rlz in gsim_lt:  # Loop over GSIM Logic_tree
                        gsim_model, gsim_weight = parser.get_value_and_weight_from_gsim_rlz(gsim_rlz)

                        # Distribute ARE:
                        pt_weight = weight*gsim_weight
                        args = (self, pt, gsim_model, pt_weight, lnSA)
                        args_list.append(args)
                        #smap.submit(args)

            are = 0
            for value in Starmap(self.pt_src_are.__func__, args_list):
                are += value
        return are


    def poe(self, lnSA):
        """
        Returns the vector-valued probability of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        are = self.are(lnSA)
        return 1-np.exp(-are*self.oqparam.investigation_time)

    def poe_parallel(self, lnSA):
        """
        Returns the vector-valued probability of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        are = self.are_parallel(lnSA)
        return 1-np.exp(-are*self.oqparam.investigation_time)

    def hazard_matrix(self, quantity='poe'):
        """
        Compute exhaustively the full VPSHA hazard matrix of ARE/POE over the N-dimensional space of
        spectral periods or parameters.
        NOTE: Parallelization occurs on the loop over seismic point sources
        WARNING !! This computation can be extremely expensive for high-dimensional problems !
        """
        # Initialization step:
        #hc_calc_method = getattr(self, quantity)  # self.poe or self.are method
        hc_calc_method = getattr(self, quantity+'_parallel')  # self.poe or self.are method
        shape = (len(self.sites),) + tuple(len(self.oqparam.imtls[str(p)]) for p in self.periods)
        max_nb = np.prod(shape)
        logging.warning('\nVPSHA matrix shape: [N_sites x N period_1 x ... x N period_k]: {}\n'.format(shape))
        logging.warning('VPSHA matrix  matrix has {} elements'.format(max_nb))

        output = np.empty(shape)
        acc_discretization = [np.log(self.oqparam.imtls[str(p)]) for p in self.periods]

        # create a N-dimensional mesh of spectral acceleration values:
        acc_meshes = np.meshgrid(*acc_discretization, indexing='ij', copy=False)
        nelts = int(np.prod(shape[1:]))  # Number of N-D pseudo spectral values
        for i in range(nelts):
            indices = np.unravel_index(i, shape[1:])  # Flat to multi-dimensional index
            accels = [ acc_meshes[j][indices] for j in range(self.ndims)]

            # Call hazard curve computation method:
            hazard_output = hc_calc_method(accels)
            # Sort results for each site:
            for k in range(len(hazard_output)):
                # Loop on sites, i.e. 1st dimension of  "hazard_output":
                output[(k,) + indices] = hazard_output[k]

        self.hc.hazard_matrix = output
        return self.hc

    def hazard_matrix_parallel(self, quantity='poe'):
        """
        Compute exhaustively the full VPSHA hazard matrix of ARE/POE over the N-dimensional space of
        spectral periods or parameters.
        NOTE: Parallelization occurs on the loop over N-D hazard matrix cells
        WARNING !! This computation can be extremely expensive for high-dimensional problems !
        """
        # Initialization step:
        hc_calc_method = getattr(self, quantity)  # self.poe or self.are method
        args = list()

        shape = (len(self.sites),) + tuple(len(self.oqparam.imtls[str(p)]) for p in self.periods)
        max_nb = np.prod(shape)

        logging.warning('\nVPSHA matrix shape: [N_sites x N period_1 x ... x N period_k]: {}\n'.format(shape))
        logging.warning('VPSHA matrix  matrix has {} elements'.format(max_nb))

        acc_discretization = [np.log(self.oqparam.imtls[str(p)]) for p in self.periods]

        # create a N-dimensional mesh of spectral acceleration values:
        acc_meshes = np.meshgrid(*acc_discretization, indexing='ij', copy=False)
        nelts = int(np.prod(shape[1:]))  # Number of N-D pseudo spectral values
        for i in range(nelts):
            indices = np.unravel_index(i, shape[1:])  # Flat to multi-dimensional index
            accels = [ acc_meshes[j][indices] for j in range(self.ndims)]
            logging.debug(f"  # Current acceleration vector: {tuple(str(p) for p in self.periods)} =  {accels}\n")

            # Call hazard curve computation method:
            #hazard_output = hc_calc_method(accels)
            args.append((indices, hc_calc_method, accels))

            """
            # Sort results for each site:
            for k in range(len(hazard_output)):  
                # Loop on sites, i.e. 1st dimension of  "hazard_output"
                # indx = np.ravel_multi_index((k,)+indices, shape)
                # indices = np.unravel_index(indx, shape)
                output[(k,) + indices] = hazard_output[k]
            """

        output = np.empty(shape)
        for result in Starmap(_matrix_cell_worker, args):
            # Sort results for each site:
            for k in range(len(result['output'])):
                # Loop on sites, i.e. 1st dimension of  "hazard_output"
                # indx = np.ravel_multi_index((k,)+indices, shape)
                # indices = np.unravel_index(indx, shape)
                output[(k,) + result['indices']] = result['output'][k]


        self.hc.hazard_matrix = output
        return self.hc


    def find_matching_vector_sample(self, target, quantity='poe', tol=None, nsol=1):
        """
        Returns a list of vector-valued coordinates corresponding to the Multi-Dimensional Hazard
        Curve ARE/POE value TARGET (within tolerance interval +/- TOL). This list of
        coordinates is obtained using an optimization algorithm

        :return: Coordinate of vector-sample with matching QUANTITY=TARGET
        """

        # TOL: Tolerance on cost-function evaluation w/r to TARGET:
        if tol is None:
            tol = target/1E3

        lower_bound = [np.log(min(self.oqparam.imtls[str(p)])) for p in self.periods]
        upper_bound = [np.log(max(self.oqparam.imtls[str(p)])) for p in self.periods]

        coord = np.empty( (nsol, 3+len(self.periods)) )
        # coord[i,:] = [ ARE_OR_POE, N_ITER, N_FEV, SA_1, ..., SA_N]
        smap = Starmap(_root_finding_worker)
        #worker_args = (getattr(self, quantity), target, lower_bound, upper_bound, tol)
        for i in range(nsol):
            #smap.submit(worker_args)
            smap.submit(getattr(self, quantity), target, lower_bound, upper_bound, tol)
        i = 0
        for res in smap:
            logging.info('{}/{}: Convergence met for sample {} ({}={})'.format(
                         i+1,nsol,np.exp(res.x),quantity,res.fun+target))
            coord[i, 0] = res.fun+target  # Evaluate ARE/POE at solution
            coord[i, 1] = res.nit
            coord[i, 2] = res.nfev
            coord[i, 3:] = np.exp(res.x)  # Convert lnSA to SA in units of g
            i = i + 1
        return coord


def _matrix_cell_worker(indices, fun, lnSA, monitor):
    result = {'indices': indices}
    result.update({'output': fun(lnSA)})
    return result


def _root_finding_worker(fun, target, lb, ub, ftol, monitor):

    def _cost_function(x, *args):
        #func = args[0]
        #target = args[1]
        #low = args[2]
        #upp = args[3]
        #if np.any(x<low) or np.any(x>upp):
        if np.any(x<lb) or np.any(x>ub):
            cost = 1E6
        else:
            cost = np.abs(target-fun(x))
        return cost

    while True:
        x0 = np.array([ np.random.uniform(l,0.3*u) for l,u in zip(lb,ub) ])
        res = minimize(_cost_function,
                       x0,
                       method='Nelder-Mead',
                       options={'fatol': ftol, 'disp': False})
        qsol = _cost_function(res.x)
        #qsol = _cost_function(res.x, fun, target, lb, ub)
        #               args=(fun, target, lb, ub),
        if res.success and (qsol<ftol):
            logging.debug('Solution found')
            logging.debug(res.message)
            break
        else:
            logging.debug('Convergence not met or function evaluation not matching target')
            logging.debug(res.message)
    return res