import parser, mdhc
from plotutils import ProgressBar
from openquake.commonlib.readinput import get_gsim_lt, get_source_model_lt
import numpy as np

class VectorValuedCalculator():
    def __init__(self, oqparam, sites_col, correlation_model):
        self.oqparam = oqparam
        self.ssm_lt = get_source_model_lt(oqparam) # Read the SSC logic tree
        self.hc = mdhc.MultiDimensionalHazardCurve(oqparam.imtls,
                                                   sites_col, correlation_model,
                                                   oqparam.maximum_distance,
                                                   oqparam.truncation_level)
        self.ndims = len(oqparam.imtls.keys())
        self.periods = oqparam.imtls.keys()
        self.sites = sites_col
        self.cm = correlation_model

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

    def are(self, *lnSA, verbose=False):
        """
        Returns the vector-valued annual rate of exceedence

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        are = 0
        for rlz in self.ssm_lt:  # Loop over realizations
            if verbose:
                print(rlz)  # Info
            _, weight = parser.get_value_and_weight_from_rlz(rlz)
            srcs = parser.get_sources_from_rlz(rlz, self.oqparam, self.ssm_lt)

            for src in srcs:  # Loop over seismic sources (area, fault, etc...)

                for pt in src:  # Loop over point-sources

                    gsim_lt = get_gsim_lt(self.oqparam, trts=[src.tectonic_region_type])
                    for gsim_rlz in gsim_lt:  # Loop over GSIM Logic_tree
                        gsim_model, gsim_weight = parser.get_value_and_weight_from_gsim_rlz(
                            gsim_rlz)

                        # Compute ARE:
                        are += self.hc.are(pt, gsim_model, *lnSA) * weight * gsim_weight
        return are

    def poe(self, *lnSA, verbose=False):
        """
        Returns the vector-valued probability of exceedance

        param *lnSA: tuple, natural logarithm of acceleration values, in unit of g.
        """
        are = self.are(*lnSA, verbose=verbose)
        return 1-np.exp(-are*self.oqparam.investigation_time)

    def hazard_matrix(self, quantity='poe', show_progress=True):
        """
        Compute exhaustively the full VPSHA hazard matrix of ARE/POE over the N-dimensional space of
        spectral periods or parameters.
        WARNING !! This computation can be extremely expensive for high-dimensional problems !
        """
        # Initialization step:
        hc_calc_method = getattr(self, quantity)
        shape = (len(self.sites),) + tuple(len(self.oqparam.imtls[p]) for p in self.periods)
        print(f'\nVPSHA matrix shape: [N_sites x N period_1 x ... x N period_k]: {shape}\n')
        output = np.empty(shape)
        if show_progress:
            pbar = ProgressBar()
        acc_discretization = [np.log(self.oqparam.imtls[p]) for p in self.periods]

        # create a N-dimensional mesh of spectral acceleration values:
        acc_meshes = np.meshgrid(*acc_discretization, indexing='ij', copy=False)
        nelts = int(np.prod(shape[1:]))  # Number of N-D pseudo spectral values
        for i in range(nelts):
            if show_progress:
                pbar.update(i, nelts, title='VPSHA computation progress: ', nsym=30)
            indices = np.unravel_index(i, shape[1:])  # Flat to multi-dimensional index
            accels = [ acc_meshes[j][indices] for j in range(self.ndims)]
            #print(f"  # Current acceleration vector: a{tuple(str(p) for p in self.periods)} =
            # {accels}\n")

            # Call hazard curve computation method:
            hazard_output = hc_calc_method(accels)
            # Sort results for each site:
            for k in range(len(hazard_output)):  # Loop on sites, i.e. 1st dimension of  "hazard_output"
                # indx = np.ravel_multi_index((k,)+indices, shape)
                # indices = np.unravel_index(indx, shape)
                output[(k,) + indices] = hazard_output[k]

        return output

