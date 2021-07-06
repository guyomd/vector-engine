import logging
from vectorengine.lib import parser, mdhc
import numpy as np
from scipy.stats import mvn
from scipy.optimize import minimize, Bounds

from openquake.baselib.parallel import Starmap

from openquake.commonlib import readinput 
from openquake.commonlib.datastore import hdf5new

from openquake.hazardlib.contexts import ContextMaker, RuptureContext
from openquake.hazardlib.calc.filters import SourceFilter
from openquake.hazardlib import const



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

    def __init__(self, oqparam, csm, sitecol, cm):
        self.oqparam = oqparam
        self.csm = csm  # Composite source model
        self.ssm_lt = csm.full_lt.source_model_lt  # SSM logic tree
        self.hc = mdhc.MultiDimensionalHazardCurve(oqparam.imtls,
                                                   sitecol,
                                                   cm,
                                                   oqparam.maximum_distance)
        self.ndims = len(oqparam.imtls.keys())
        self.periods = readinput.get_imts(oqparam)
        self.sites = sitecol
        self.cm = cm
        self.srcfilter = SourceFilter(sitecol, oqparam.maximum_distance)
        self.integration_prms = {'truncation_level': oqparam.truncation_level,
                                 'abseps': 0.0001,  # Documentation: Optimal value is 1E-6
                                 'maxpts': self.ndims*10  # Documentation: Optimal value is len(lnSA)*1000
                                }
        self.integration_prms.update({'trunc_norm': self._truncation_normalization_factor()})


    def _truncation_normalization_factor(self):
        """
        Returns the N-D normalization factor for the normal law integration
        over the [ -n*std, +n*std ] domain
        """
        mu = np.zeros((self.ndims,))
        lower_trunc = -self.integration_prms['truncation_level']*np.ones_like(mu)
        upper_trunc = self.integration_prms['truncation_level']*np.ones_like(mu)
        trunc_norm, _ = mvn.mvnun(lower_trunc,
                                  upper_trunc,
                                  mu,
                                  self.hc.corr,
                                  abseps=self.integration_prms['abseps'],
                                  maxpts=self.integration_prms['maxpts'])
        return trunc_norm


    def _get_sources_from_smr(self, smr):
        sources = []
        groups = self.csm.get_groups(smr)
        for grp in groups:
            for srcs, sitecol in self.srcfilter.filter(grp):
                for src in srcs:
                    sources.append(src)
        return sources


    def count_sources(self):
        return len(self.csm.get_sources())


    def count_pointsources(self):
        """
        Returns the total number of point-sources involved in the current calculation
        """
        n = 0
        # Loop on source model logic-tree realizations:
        for smr, rlzs in self.csm.full_lt.get_rlzs_by_smr().items():
            srcs = self._get_sources_from_smr(smr)
            for src in srcs:
                for _ in self.srcfilter.filter(src):
                    n += 1
        return n


    def gm_poe(self, gsim, dist_ctx, rup_ctx, site_ctx, lnSA):
        """
        Returns the multivariate probability to exceed a specified set of
        ground motion acceleration levels
        """
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

        mu = np.zeros((self.ndims,))
        prob = np.zeros((nsites,))
        for j in range(nsites):
            # Build covariance matrix:
            #D = np.diag(lnSTD[j,:])
            #cov = D @ self.hc.corr @ D
            #lower_trunc = lnAVG[j,:]-self.integration_prms['truncation_level']*np.sqrt(np.diag(cov))
            #upper_trunc = lnAVG[j,:]+self.integration_prms['truncation_level']*np.sqrt(np.diag(cov))
            lower_trunc = -self.integration_prms['truncation_level']*np.ones_like(mu)
            upper_trunc = self.integration_prms['truncation_level']*np.ones_like(mu)
            lower = (np.array(lnSA) - lnAVG[j,:])/lnSTD[j,:] # Convert accel to epsilon
            
            if np.any(lower >= upper_trunc):
                # Requested value is above the 3-sigma truncature for at least one spectral period:
                prob[j] = 0
                continue

            if np.any(lower <= lower_trunc):
                indices = (lower <= lower_trunc).nonzero()
                lower[indices] = -self.integration_prms['truncation_level']
          
            prob[j], error = mvn.mvnun(lower,
                                  upper_trunc,
                                  mu,
                                  self.hc.corr,
                                  abseps=self.integration_prms['abseps'],
                                  maxpts=self.integration_prms['maxpts'])
      
            # Normalize poe over the truncation interval [-n*sigma, n*sigma]
            prob[j] /= self.integration_prms['trunc_norm']

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
            rup_ctx, site_ctx, dist_ctx = context_maker.make_contexts(self.sites, r)
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
        # Loop on source model logic-tree realizations:
        for smr, rlzs in self.csm.full_lt.get_rlzs_by_smr().items():
            srcs = self._get_sources_from_smr(smr)
            
            # Loop on ground-motion model logic-tree realizations:
            for r in rlzs:
                weight = r.weight
                gsim_model, _ = \
                        parser.get_value_and_weight_from_gsim_rlz(r.gsim_rlz)
        
                # Loop over (filtered) seismic sources (area, fault, etc...)
                for src in srcs: 

                    # Loop over point-sources:
                    for pt, _ in self.srcfilter.filter(src):  
                        are += self.pt_src_are(pt, gsim_model, weight['weight'], lnSA, None)
        return are


    def are_parallel(self, lnSA):
        """
        Returns the vector-valued annual rate of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        args_list = list()

        # Loop on source model logic-tree realizations:
        for smr, rlzs in self.csm.full_lt.get_rlzs_by_smr().items():
            srcs = self._get_sources_from_smr(smr)
            
            # Loop on ground-motion model logic-tree realizations:
            for r in rlzs:  # Loop over realizations
                weight = r.weight 
                gsim_model, _ = \
                        parser.get_value_and_weight_from_gsim_rlz(r.gsim_rlz)
    
                # Loop over (filtered) seismic sources (area, fault, etc...)
                for src in srcs:  
 
                    # Loop over point-sources:
                    for pt, _ in self.srcfilter.filter(src):  
                            # Distribute ARE:
                            args = (self, pt, gsim_model, weight['weight'], lnSA)
                            args_list.append(args)
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


    def hazard_matrix_calculation(self, quantity='poe'):
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
        logging.warning('hazard matrix shape: [N_sites x N_IMT_1 x ... x N_IMT_k]: {}'.format(shape))
        logging.warning('hazard matrix has {} elements'.format(max_nb))

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


    def hazard_matrix_calculation_parallel(self, quantity='poe'):
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

        logging.warning('hazard matrix shape: [N_sites x N_IMT_1 x ... x N_IMT_k]: {}'.format(shape))
        logging.warning('hazard matrix has {} elements'.format(max_nb))

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


    def find_matching_poe_parallel_runs(self, target, quantity='poe', tol=None, nsol=1, outputfile=None):
        """
        Returns a list of vector-valued coordinates corresponding to the Multi-Dimensional Hazard
        Curve ARE/POE value TARGET (within tolerance interval +/- TOL). This list of
        coordinates is obtained using an optimization algorithm. Parallelization is realized by
        sending one individual optimization run on each worker.

        :return: Coordinate of vector-sample with matching QUANTITY=TARGET
        """

        # TOL: Tolerance on cost-function evaluation w/r to TARGET:
        if tol is None:
            tol = target/1E3

        lower_bound = [np.log(min(self.oqparam.imtls[str(p)])) for p in self.periods]
        upper_bound = [np.log(max(self.oqparam.imtls[str(p)])) for p in self.periods]

        coord = np.empty( (nsol, 3+len(self.periods)) )
        # coord[i,:] = [ ARE_OR_POE, N_ITER, N_FEV, SA_1, ..., SA_N]
        worker_args = list() 
        for i in range(nsol):
            rs = np.random.RandomState(seed=np.random.random_integers(0,1E9))
            worker_args.append((getattr(self, quantity), target, lower_bound, upper_bound, tol, rs))
        i = 0
        for res in Starmap(_root_finder_worker, worker_args):
            logging.info('Starting point: {}'.format(res.x0))
            logging.info('{}/{}: Convergence met for sample {} ({}={})'.format(
                         i+1,nsol,np.exp(res.x),quantity,res.fun+target))
            coord[i, 0] = res.fun+target  # Evaluate ARE/POE at solution
            coord[i, 1] = res.nit
            coord[i, 2] = res.nfev
            coord[i, 3:] = np.exp(res.x)  # Convert lnSA to SA in units of g
            i = i + 1
        with open(outputfile, 'ab') as f:
            np.savetxt(f, coord, fmt='%.6e', delimiter=',')


    def find_matching_poe(self, target, quantity='poe', tol=None, nsol=1, outputfile=None):
        """
        Returns a list of vector-valued coordinates corresponding to the Multi-Dimensional Hazard
        Curve ARE/POE value TARGET (within tolerance interval +/- TOL). This list of
        coordinates is obtained using an optimization algorithm. Parallelization is
        realized by distributing individual AREs over each point source.

        :return: Coordinate of vector-sample with matching QUANTITY=TARGET
        """

        # TOL: Tolerance on cost-function evaluation w/r to TARGET:
        if tol is None:
            tol = target/1E3

        lower_bound = [np.log(min(self.oqparam.imtls[str(p)])) for p in self.periods]
        upper_bound = [np.log(max(self.oqparam.imtls[str(p)])) for p in self.periods]

        coord = np.empty( (nsol, 3+len(self.periods)) )
        # NB: coord[i,:] = [ ARE_OR_POE, N_ITER, N_FEV, SA_1, ..., SA_N]
        hc_calc_method = getattr(self, quantity+'_parallel')
        for i in range(nsol):
            rs = np.random.RandomState(seed=np.random.random_integers(0,1E9))
            res = _root_finder_worker(hc_calc_method, target, lower_bound, upper_bound, tol, rs, None)
            logging.info('Starting point: {}'.format(res.x0))
            logging.info('{}/{}: Convergence met for sample {} ({}={})'.format(
                i + 1, nsol, np.exp(res.x), quantity, res.fun + target))
            coord[i, 0] = res.fun + target  # Evaluate ARE/POE at solution
            coord[i, 1] = res.nit
            coord[i, 2] = res.nfev
            coord[i, 3:] = np.exp(res.x)  # Convert lnSA to SA in units of g
            with open(outputfile, 'ab') as f:
                np.savetxt(f, coord[i,:][np.newaxis,:], fmt='%.6e', delimiter=',')


def _matrix_cell_worker(indices, fun, lnSA, monitor):
    result = {'indices': indices}
    result.update({'output': fun(lnSA)})
    return result


def _root_finder_worker(fun, target, lb, ub, ftol, rs, monitor):

    def _cost_function(x):
        if np.any(x<lb) or np.any(x>ub):
            cost = 1E6
        else:
            cost = np.abs(target-fun(x))
        return cost

    while True:
        x0 = np.array([ rs.uniform(l,u) for l,u in zip(lb,ub) ])
        res = minimize(_cost_function,
                       x0,
                       method='Nelder-Mead',
                       options={'fatol': ftol, 'disp': False})
        qsol = _cost_function(res.x)
        res.x0 = x0
        if res.success and (qsol<ftol):
            logging.debug('Solution found')
            logging.debug(res.message)
            break
        else:
            logging.debug('Convergence not met or function evaluation not matching target')
            logging.debug(res.message)
    return res
