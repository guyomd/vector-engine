import numpy as np
# Note : Useful function for multidimensional matrix integration:
# np.ndindex() iterator ovre multi_index for each matrix element
# np.apply_over_axes()
from scipy.integrate import simps
from copy import deepcopy


def _integration1D(m, x, axis_indx):
    a = simps(m, x[axis_indx], axis=axis_indx)
    return a


def integrationND(arr, x):
    func = lambda mat, ax: _integration1D(mat, x, ax)
    integ = np.apply_over_axes(func, arr, list(range(len(arr.shape)))).squeeze()
    return integ


def poe2pdf(m, x, diff_option='diff'):
    ndim = len(m.shape)
    pdf = deepcopy(m)
    shape = np.array(m.shape)
    for k in range(ndim):
        if diff_option == 'gradient':
            pdf = -np.gradient(pdf, x[k], axis=k, edge_order=1)
            #x[k][0] = x[k][0]+0.5*(x[k][1]-x[k][0])
            #x[k][-1] = x[k][-2]+0.5*(x[k][-1]-x[k][-2])
        elif diff_option == 'diff':
            shape[k] -= 1
            xmat = np.meshgrid(*x, indexing='ij')[k]
            pdf = -np.diff(pdf, 1, axis=k) / np.diff(xmat, 1, axis=k)
            xx = list()
            for j in range(ndim):
                if j == k:
                    xx.append(np.array(x[k][0:-1] + 0.5 * np.diff(x[k])))
                else:
                    xx.append(x[j])
            x = xx
        else:
            raise ValueError(f'Unrecognized "diff_option" value: {diff_option}')
    return pdf, x


def cdf2pdf(m, x, diff_option='diff', indexing='ij'):
    ndim = len(m.shape)
    pdf = deepcopy(m)
    shape = np.array(m.shape)
    for k in range(ndim):
        if diff_option == 'gradient':
            pdf = np.gradient(pdf, x[k], axis=k, edge_order=1)
            #x[k][0] = x[k][0]+0.5*(x[k][1]-x[k][0])
            #x[k][-1] = x[k][-2]+0.5*(x[k][-1]-x[k][-2])
        elif diff_option == 'diff':
            shape[k] -= 1
            xmat = np.meshgrid(*x, indexing=indexing)[k]
            pdf = np.diff(pdf, 1, axis=k) / np.diff(xmat, 1, axis=k)
            xx = list()
            for j in range(ndim):
                if j == k:
                    xx.append(np.array(x[k][0:-1] + 0.5 * np.diff(x[k])))
                else:
                    xx.append(x[j])
            x = xx
        else:
            raise ValueError(f'Unrecognized "diff_option" value: {diff_option}')
    return pdf, x


def pdf2cdf(m, x):
    ndim = len(m.shape)
    for k in range(ndim):
        #dx = np.array([0, np.diff(x[k])])
        m = np.cumsum(m, axis=k)
    return m


def pdf2poe(m, x):
    ndim = len(m.shape)
    for k in range(ndim):
        #dx = np.array([0, np.diff(x[k])])
        m = np.flip(np.cumsum(np.flip(m, axis=k), axis=k), axis=k)
    return m


def marginals1D(pdf, x, axis=None):
    """
    Compute the set of N 1-D marginal probability functions associated
    with the input N-D probability density function.
    Note: if axis is None, marginals are computed for each axis, and output
    in the same order than pdf dimensions. If axis is a list, then marginals
    are output in the list order.
    """

    def integ1D(mat, ax):
        return _integration1D(mat, x, ax)

    ndims = len(pdf.shape)
    all_axes = list(range(ndims))
    if axis is None:
        axis = all_axes
    elif isinstance(axis, int):
        axis = [axis]
    marginals = list()
    for a in axis:
        integration_axes = list(filter(lambda x: x != a, all_axes))
        # integration_axes = all_axes[:a]+all_axes[a+1:]  # Equivalent explicit form
        marg = np.apply_over_axes(integ1D, pdf, np.flipud(integration_axes)).squeeze()
        print(f'Integral of 1-D marginal along axis {a}: {simps(marg, x[a])}')
        marginals.append(marg)
    return marginals


def pdf2poe1D(pdf, x):
    poe = np.zeros((len(x),))
    for i in range(len(x)):
        poe[i] = simps(pdf[i:], x[i:])
    return poe


def build_marginals(mat, imtls, normalize=False):
    """
    Builds the set of N unidimensional marginal PDF from a N-D hazard matrix

    :param mat: N-D numpy.ndarray, N-D hazard matrix expressed in terms of POE
    :param imtls: Dictionary of IM values. Each key stands for an IMT, and the
                  corresponding value is a list of IM values.
    :param normalize: logical, specify whether N-D PDF should be normalized (default: False)

    """
    # Remove dimensions of length 1 from the hazard matrix:
    mat = np.squeeze(mat) 
    # Compute log of abscissa for each dimension:
    logx = np.array([np.log(imtls[p]) for p in imtls.keys()])
    # x = np.log(np.array([imtls[p] for p in imtls.keys()]))
    nd = len(mat.shape)
    print(f'Maximum value of the {nd}-D POE hazard matrix : {mat.max()}')
    pdf, xmod = poe2pdf(mat, logx, diff_option='gradient')
    sumpdf = integrationND(pdf, xmod)
    print(f'Sum of {nd}-D PDF : {sumpdf}')
    if normalize:
        pdf = pdf / sumpdf
        print(f'Sum after normalization : {integrationND(pdf, xmod)}')

    marginals_pdf = marginals1D(pdf, xmod)
    marginals_poe = list()
    for k in range(len(marginals_pdf)):
        marginals_poe.append(pdf2poe1D(marginals_pdf[k], xmod[k]))

    xupd = np.array([np.exp(arr) for arr in xmod])
    return marginals_poe, marginals_pdf, xupd
