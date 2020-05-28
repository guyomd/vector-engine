
import numpy as np

from openquake.hazardlib import imt


class MultiDimensionalHazardCurve():
    """
    Multi-dimensional Hazard Curve container
    """
    def __init__(self, imtls, sites, gmcm, maximum_distance):
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
        self.hazard_matrix = None


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


    def plot(self):
        pass