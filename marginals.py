import h5py
import numpy as np
# Note : Useful function for multidimensional matrix integration:
# np.ndindex() iterator ovre multi_index for each matrix element
# np.apply_over_axes()

from scipy.integrate import cumtrapz, simps
from copy import deepcopy
from openquake.commonlib.readinput import get_oqparam
from matplotlib import pyplot as plt
from plotutils import plot_matrix

def read_hzd_curve(hdf5file):
    with h5py.File(hdf5file, 'r') as f:
        allsites_mat = f.get('hazard_matrix')[()]
        # Matrix shape is [Nsites, Nper1, ..., Nperk):
        assert allsites_mat.shape[0] == 1 # Only one site permitted
        return np.squeeze(allsites_mat)


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
            pdf = -np.gradient(pdf, x[k], axis=k)
        elif diff_option == 'diff':
            shape[k] -= 1
            xmat = np.meshgrid(*x, indexing='ij')[k]
            pdf = -np.diff(pdf, 1, axis=k)/np.diff(xmat, 1, axis=k)
            xx = list()
            for j in range(ndim):
                if j==k:
                    xx.append(np.array(x[k][0:-1]+0.5*np.diff(x[k])))
                else:
                    xx.append(x[j])
            x = xx
        else:
            raise ValueError(f'Unrecognized "diff_option" value: {diff_option}')

        plot_matrix(pdf, x[0], x[1], f'ln {periods[0]})', f'ln {periods[1]}', f'POE integrated along axe {k}', ndigits_labels=4)
    return pdf, x


def marginals1D(pdf, x, axis=None):
    """
    Compute the set of N 1-D marginal probability functions associated
    with the input N-D probability density function.
    Note: if axis is None, marginals are computed for each axis, and output
    in the same order than pdf dimensions. If axis is a list, then marginals
    are output in the list order.
    """
    func = lambda mat, ax: _integration1D(mat, x, ax)

    ndims = len(pdf.shape)
    all_axes = list(range(ndims))
    if axis is None:
        axis = all_axes
    elif isinstance(axis, int):
        axis = [axis]
    marginals = list()
    for a in axis:
        integration_axes = list(filter(lambda x: x!=a, all_axes))
        #integration_axes = all_axes[:a]+all_axes[a+1:]  # Equivalent explicit form
        marg = np.apply_over_axes(func, pdf, np.flipud(integration_axes)).squeeze()
        print(f'Integral of 1-D marginal along axis {a}: {simps(marg, x[a])}')
        marginals.append(marg)
    return marginals


def pdf2poe1D(pdf, x):
    poe = np.zeros((len(x),))
    for i in range(len(x)):
        poe[i] = simps(pdf[i:], x[i:])
    return poe



#### MAIN PART OF SCRIPT STARTS HERE !!! ####

#h5file = "/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_4D_20200812/poe_2020-08-12T105335.hdf5"
#job_ini = '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_4D_20200812/job_4D.ini'
h5file = "/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/2D_Curve/poe_2D_1s_0.05s.hdf5"
job_ini = '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/AreaSourceClassicalPSHA/modified_job_2D.ini'

ref_hzd_curves = {
    'SA(0.05)': {
        'hdf5': '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/1D_Curves/poe_1D_0.05s.hdf5',
        'ini': '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/AreaSourceClassicalPSHA/modified_job_1D_0.05s.ini'},
    'SA(1.0)': {
        'hdf5': '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/1D_Curves/poe_1D_1s.hdf5',
        'ini': '/home/b94678/Calculs/vectorValuedPSHA/vpsha-data/test_2D/AreaSourceClassicalPSHA/modified_job_1D_1s.ini'}
}

mat = read_hzd_curve(h5file)
nd = len(mat.shape)

print(f'Maximum value of the {nd}-D POE: {mat.max()}')
oq = get_oqparam(job_ini)
imtls = oq.imtls
per = oq.imtls.keys()
periods = list(per)

# Warning: For a proper integration, log of acceleration values should be used!
#          (to be coherent with the integration space used in poe_gm() method)
logx = np.array([ np.log(imtls[p]) for p in per ])
x = np.log(np.array([ imtls[p] for p in per ]))

plot_matrix(mat, x[0], x[1], f'ln {periods[0]}', f'ln {periods[1]}', 'Prob. of exceedance', ndigits_labels=4)


pdf, xmid = poe2pdf(mat, x)
plt.show()
sumpdf = integrationND(pdf, xmid)

normalize_pdf = False
print(f'Sum of {nd}-D PDF before normalization: {sumpdf}')
if normalize_pdf:
    pdf = pdf/sumpdf
    print(f'Sum after normalization: {integrationND(pdf, xmid)}')

marg_pdf = marginals1D(pdf, xmid)
marg_poe = list()
for k in range(len(marg_pdf)):
    marg_poe.append( pdf2poe1D(marg_pdf[k], xmid[k]) )


### PLOTS ###
"""
plt.Figure()
for i in range(len(periods)):
    plt.semilogy(xmid[i],margs[i],label=str(periods[i]))
plt.legend()
plt.show()
"""
for i in range(len(periods)):
    ref_curve = read_hzd_curve(ref_hzd_curves[str(periods[i])]['hdf5'])
    oqtmp = get_oqparam(ref_hzd_curves[str(periods[i])]['ini'])
    ref_curve_x = oqtmp.imtls[periods[i]]
    ref_curve_x = np.log(ref_curve_x)
    plt.Figure()
    # Reference POE obtained by a separate VPSHA 1-D:
    plt.plot(ref_curve_x, ref_curve, label='1-D ref.')
    # 1-D marginal PDF from the N-D calculation:
    hc = plt.plot(xmid[i],marg_pdf[i], ls=':', lw = 1, label='PDF')
    # 1-D marginal POE from the N-D calculation:
    plt.plot(xmid[i],marg_poe[i], color = hc[-1].get_color(),
             ls='-', marker=None, fillstyle='full', label='POE')
    plt.xlabel('ln(PSA in g)')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.title(str(periods[i]))
    plt.legend()
    plt.grid(True)
    plt.show()


exit()


if len(pdf.shape)==2:
    plt.Figure()
    plt.imshow(pdf)
    plt.colorbar()
    plt.show()



