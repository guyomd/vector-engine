
import scipy
import numpy as np
from scipy.stats import multivariate_normal, norm, mvn
from openquake.hazardlib.contexts import ContextMaker, RuptureContext
from openquake.hazardlib import const, imt


class MultiDimensionalHazardCurve():
    """
    Multi-dimensional Hazard Curve designed for VPSHA calculations using the
    Openquake Engine
    """
    def __init__(self, imtls, sites, gmcm, maximum_distance, truncation_level):
        """
        MultiDimensionalHazardCurve initialization.

        :param imtls: DictArray, using periods as keys and acceleration levels
        as elements. See property "imtls" of class
        "openquake.commonlib.oqvalidation.OqParam"
        :param sites: instance of  class "openquake.hazardlib.site.SiteCollection"
        :param gsim: instance of Openquake GSIM class
        :param gmcm: instance of "imcm.IntensityMeasureCorrelationModel" class
        :param maximum_distance: maximum  point-source to site distance for ARE calculations
        :param truncation_level: float, truncation level of the GMPE in std. dev.
        """
        self.ndims = len(imtls.keys())
        self.periods = list()
        for per in list(imtls.keys()):
            self.periods.append(imt.from_string(per))
        self.gmcm = gmcm
        self.set_correlation_matrix()
        self.sites = sites
        self.maximum_distance = maximum_distance
        self.truncation_level = truncation_level

    def set_correlation_matrix(self):
        """
        Define the correlation matrix of acceleration for spectral period pairs
        """
        def is_pos_semidef(x):
            return np.all(np.linalg.eigvals(x) >= 0)

        corr = np.zeros((self.ndims, self.ndims))
        per = [ p.period for p in self.periods ]
        for i in range(self.ndims):
            for j in range(i, self.ndims):
                corr[i, j] = self.gmcm.rho(per[i], per[j])
                if i != j:
                    corr[j, i] = corr[i, j]
        if not is_pos_semidef(corr):
            print(f'ERROR: Correlation matrix is not positive, semi-definite')
            val = np.linalg.eigvals(corr)
            print(f'EIGENVALUES: {val}')
            print(f'ADVICE: REPLACE negative eigenvalues with 0')
            raise ValueError('Inappropriate Ground-motion Correlation matrix')
        else:
            self.corr = corr

    def poe_gm(self, gsim, dist_ctx, rup_ctx, site_ctx, *args):
        """
        Returns the multivariate probability to exceed a specified set of
        ground motion acceleration levels
        """
        lower = np.array(*args) # Natural logarithm of acceleration values
        abseps = 0.0001  # Documentation: Optimal value is 1E-6
        maxpts = len(lower)*10  # Documentation: Optimal value is len(lower)*1000
        nsites = len(site_ctx)
        lnSA = np.zeros((nsites,self.ndims))
        lnSTD = np.zeros(lnSA.shape)
        for i in range(self.ndims):
            # gsim().get_mean_and_stddevs returns ground-motion as ln(PSA) in units of g:
            means, stddevs = gsim.get_mean_and_stddevs(site_ctx,
                                                       rup_ctx,
                                                       dist_ctx,
                                                       self.periods[i],
                                                       [const.StdDev.TOTAL])
            lnSA[:,i] = np.squeeze(means)
            lnSTD[:,i] = np.squeeze(stddevs[0])
        poe = np.zeros((nsites,))
        for j in range(nsites):
            # Build covariance matrix:
            D = np.diag(lnSTD[j,:])
            cov = D @ self.corr @ D
            lower_trunc = lnSA[j,:]-self.truncation_level*np.sqrt(np.diag(cov))
            upper_trunc = lnSA[j,:]+self.truncation_level*np.sqrt(np.diag(cov))
            if np.any(lower >= upper_trunc):
                # Requested value is above the 3-sigma truncature for at least one spectral period:
                poe[j] = 0
            else:
                trunc_norm, _ = mvn.mvnun(lower_trunc,
                                   upper_trunc,
                                   lnSA[j,:],
                                   cov,
                                   abseps=abseps,
                                   maxpts=maxpts)

                poe[j], error = mvn.mvnun(lower,
                                  upper_trunc,
                                  lnSA[j,:],
                                  cov,
                                  abseps=abseps,
                                  maxpts=maxpts)
                poe[j] /= trunc_norm  # Normalize poe over the truncation interval [-n*sigma, n*sigma]
        return poe

    def are(self, pt_src, gsim, *lnSA):
        """
        Returns the Annual Rate of Exceedance for all sites given a vector of IM values

        :param pt_src: single instance of class
            "openquake.hazardlib.source.area.PointSource"
        :param gsim: tuple, containing (only one?) instance of Openquake GSIM class
        :param *lnSA: natural logarithm of acceleration values for each spectral period.
            Note : Values should be ordered in the same order than self.periods
        """
        haz = 0
        # Loop over ruptures, one rupture for each combination of (mag, nodal plane, hypocentral
        # depth):
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

            haz += r.occurrence_rate * self.poe_gm(gsim[0], dist_ctx, rup_ctx, site_ctx, *lnSA)
        return haz

    def plot(self):
        pass