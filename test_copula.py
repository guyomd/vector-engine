import numpy as np
from oqutils import results as r
import os
from matplotlib import pyplot as plt

from scipy.interpolate import interpn
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

from vectorengine.lib.imcm import BakerJayaram2008
from vectorengine.lib.marginals import cdf2pdf, integrationND, pdf2poe
from vectorengine.lib.plotutils import plot_matrix, plot_contours
from vectorengine.lib.parser import load_dataset_from_hdf5


def set_correlation_matrix(periods, imcm):
    """
    Define the correlation matrix of acceleration for spectral period pairs
    """
    def is_pos_semidef(x):
        return np.all(np.linalg.eigvals(x) >= 0)

    ndims = len(periods)
    corr = np.zeros((ndims, ndims))
    for i in range(ndims):
        for j in range(i, ndims):
            corr[i, j] = imcm.rho(periods[i], periods[j])
            if i != j:
                corr[j, i] = corr[i, j]
    if not is_pos_semidef(corr):
        val = np.linalg.eigvals(corr)
        logging.warning('Correlation matrix is not positive, semi-definite')
        logging.warning('eigenvalues: {}'.format(val))
        logging.warning('--> replace negative eigenvalues with 0')
        logging.error('Incorrect inter-IM correlation matrix')
        # raise ValueError('Inappropriate Ground-motion Correlation matrix')
    else:
        return corr


def arrays2flatmesh(*x, indexing='ij'):
    xmesh = np.meshgrid(*x, indexing=indexing)
    xflat = list()
    for i in range(len(xmesh)):
        xflat.append(xmesh[i].flatten())
    return xflat


fig, axs = plt.subplots(1,2, figsize=(12,6))

hazdir = "/home/b94678/Work/vectorValuedPSHA/vpsha-data/2021_full_haz_matrix/test_2D/AreaSourceClassicalPSHA/OUTPUTS/"
hazfiles = {'0.05': 'hazard_curve-mean_9-SA(0.05).xml',
            '1.0': 'hazard_curve-mean_9-SA(1.0).xml'}

directNDhazfile = "/home/b94678/Work/vectorValuedPSHA/vpsha-data/2021_full_haz_matrix/test_2D/poe_with_marginals_2021-04-12T164741.hdf5"

periods = [float(str(s)) for s in hazfiles.keys()]
corrmat = set_correlation_matrix(periods, BakerJayaram2008())
print('Correlation matrix:\n{}'.format(corrmat))
hazcurves = []
labels = []
for period in hazfiles.keys():
    hazcurves.append(r.HazardCurveResult(os.path.join(hazdir, hazfiles[period])).curves[0])
    labels.append(period)


def plot_curves(xs, ys, ax, labels=None, xlabel=None, ylabel=None, title=None):
    for x,y, lab in zip(xs, ys, labels):
        ax.loglog(x, y, label=lab)
    opts = ['xlabel', 'ylabel', 'title']
    for opt in opts:
        if eval(opt) is None:
            pass
        else:
            fun = getattr(ax, 'set_'+opt)
            fun(eval(opt))
    if labels is not None:
        ax.legend()
    return ax


imls = [c.iml for c in hazcurves]
poes = [c.poe for c in hazcurves]
# PLOT HAZARD CURVES (SURVIVAL):
plot_curves(imls,
            poes,
            axs[0],
            labels=labels,
            title="Hazard curves",
            xlabel='IML',
            ylabel='Pr[X$\geq$IML]')

# PLOT CDFs:
cdfs = [1-c.poe for c in hazcurves]
plot_curves(imls,
            cdfs,
            axs[1],
            labels=labels,
            title="Hazard CDF",
            xlabel='IML',
            ylabel='Pr[X$\leq$IML]')


class InterpolatedHazardCDF(object):
    def __init__(self, cdf, imls):
        self.binned_cdf = cdf
        self.binned_imls = imls
        self.binned_pdf = np.gradient(cdf, imls)

    def cdf(self, x):
        return np.interp(np.log(x),
                         np.log(self.binned_imls),
                         self.binned_cdf,
                         left=0.0, right=1.0)

    def pdf(self, x):
        return np.interp(np.log(x),
                         np.log(self.binned_imls),
                         self.binned_pdf,
                         left=0.0, right=0.0)


class GaussianCopulaModel(object):
    def __init__(self, rho):
        self.rho = np.array(rho)
        self.det = np.linalg.det(rho)
        self.ndim = self.rho.shape[0]
        self._C = np.linalg.inv(self.rho)-np.identity(self.ndim)
        self._K = 1/np.sqrt(self.det)
        self._prec = np.float64(1E-15)
        self._ubounds = np.array([ self._prec, 1.0-self._prec ])


    def cdf(self, u):
        return mvn(cov=self.rho).cdf(norm.ppf(u))
        pass


    def pdf(self, u):
        u = np.reshape(u,(self.ndim,1))
        u[ u <= self._ubounds[0] ] = self._ubounds[0]  # Prevent values too close to 0.0 (U --> -inf)
        u[ u >= self._ubounds[1] ] = self._ubounds[1]  # Prevent values too close to 1.0 (U --> +inf)
        P = norm.ppf(u)  # Inverse standard normal CDF
        return self._K*np.exp(-0.5*np.matmul(np.matmul(P.T,self._C),P))


# Transform N-dinmensional axes coordinates into N-D variables bounded in the [0, 1]^N domain:
U = list()
f = list()
ndim = len(imls)
for i in range(ndim):
    F = InterpolatedHazardCDF(cdfs[i], imls[i])  # Note: here imls[i] are X-ordinates from the original hazard curve
    U.append(F.cdf(imls[i])) 
    f.append(F.pdf(imls[i])) 

fi = np.meshgrid(*f, indexing='ij')
Ui = np.meshgrid(*U, indexing='ij')
u_shp = Ui[0].shape
u_size = Ui[0].size

# Compute Copula model:
cop = GaussianCopulaModel(corrmat)
C = np.zeros_like(Ui[0])
c = np.zeros_like(Ui[0])
for k in range(u_size):
    i = np.unravel_index(k, u_shp)
    sample = np.array([Ui[j][i] for j in range(ndim)])
    C[i] = cop.cdf(sample)  # Multivariate cumulative distribution function
    pdf_product = np.prod(np.array([fi[j][i] for j in range(ndim)]))
    c[i] = cop.pdf(sample)*pdf_product  # Multivariate density function


print(f'Max. of C = {C.max()}')
print(f'Max. of c = {c.max()}')
PDF, X_PDF = cdf2pdf(C, imls, indexing='ij')  # Multivariate PDF
#PDF = c
#X_PDF = imls
integ = integrationND(PDF, X_PDF)
PDF = PDF #/integ
print(f'Integral of PDF = {integ}')
POE = pdf2poe(PDF, X_PDF)  # Multivariate POE
POE = POE/POE.max()
print(f'Max. of POE = {POE.max()}')
# Plot PDF:
plot_contours(PDF, np.log10(X_PDF[0]), np.log10(X_PDF[1]),
              'log10 SA({} s.) in g'.format(periods[0]),
              'log10 SA({} s.) in g'.format(periods[1]),
              'PDF', 
              cmap='viridis') 

#plot_matrix(PDF, X_PDF[1], X_PDF[0],
#            f'SA({periods[0]} s.) in g',
#            f'SA({periods[1]} s.) in g',
#            'PDF')

### Plot POE and compare with direct calculation:
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
#plot_matrix(POE, X_PDF[1], X_PDF[0],
#            f'SA({periods[1]} s.) in g',
#            f'SA({periods[0]} s.) in g',
#            'POE - copula method', ax=axs[0])
plot_contours(POE, np.log10(X_PDF[1]), np.log10(X_PDF[0]),
            f'log10 SA({periods[1]} s.) in g',
            f'log10 SA({periods[0]} s.) in g',
            'POE - copula method', 
            ax=axs[0],
            cmap='viridis')

# Direct N-D hazard calculation:
nd_data, nd_imtls = load_dataset_from_hdf5(directNDhazfile)
print(f'Max of direct N-D POE: {nd_data.max()}')
#plot_matrix(nd_data, 
#            nd_imtls[f'SA({periods[1]})'],
#            nd_imtls[f'SA({periods[0]})'],
#            f'SA({periods[1]})',
#            f'SA({periods[0]})',
#            'POE - direct method', ax=axs[1])
plot_contours(nd_data, 
              np.log10(nd_imtls[f'SA({periods[1]})']),
              np.log10(nd_imtls[f'SA({periods[0]})']),
              f'log10 SA({periods[1]} s.) in g',
              f'log10 SA({periods[0]} s.) in g',
              'POE - direct method', 
              ax=axs[1],
              cmap='viridis')

# Evaluate (interpolated) "direct" 2-D hazard matrix at samples EVAL_AT:
x_direct = nd_imtls[f'SA({periods[0]})']
y_direct = nd_imtls[f'SA({periods[1]})']
#print(nd_imtls)

points = np.array([np.log(x_direct), np.log(y_direct)])
xi_flat = arrays2flatmesh(*X_PDF)
xi = np.array([xi for xi in zip(*xi_flat)])
xi = np.log(xi)

interpPOE = interpn(points,
                    nd_data, 
                    xi,
                    method='linear',
                    bounds_error=False,
                    fill_value=None)
newshape = tuple(len(x) for x in X_PDF)
interpPOE = np.reshape(interpPOE, newshape)

plot_matrix(interpPOE-POE, X_PDF[1], X_PDF[0],
            f'SA({periods[1]} s.) in g',
            f'SA({periods[0]} s.) in g',
            'Difference (direct - copula)', 
            ax=axs[2])

plt.show()
