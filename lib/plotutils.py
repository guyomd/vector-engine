import sys
import os
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from vengine.lib.parser import get_matrix_values_and_axes, read_hzd_matrix
from vengine.lib.marginals import build_marginals, poe2pdf
from openquake.commonlib.readinput import get_oqparam


def save_plot(filename, h=None, size=(8,6), keep_opened=False, tight=False):
    if isinstance(h, plt.Axes):
        h = h.get_figure()  # Eventually, retrieve parent Figure handle
    elif isinstance(h, plt.Figure):
        pass
    elif h is None:
        print('INFO: save_plot() method will save the current figure')
        h = plt.gcf()
    plt.figure(h.number)   # Set as current figure
    h.set_size_inches(size[0], size[1])  # size = (width, height)
    kwargs = dict()
    if tight:
        kwargs.update({'bbox_inches': 'tight'})
    else:
        kwargs.update({'bbox_inches': None})
    plt.savefig(filename, dpi=150, format='png', pad_inches=0.05, **kwargs)
    print(f'Figure saved in {filename}')
    if not keep_opened:
        plt.close(h.number)


def plot_matrix(M, x, y, xlabel, ylabel, title, cbar_title=None, ndigits_labels=2):
    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    n = len(x)
    plt.figure()
    plt.imshow(M, aspect='auto',
               interpolation="none")
    ax = plt.gca()
    x_str = [('{:.'+str(ndigits_labels)+'f}').format(a) for a in x]
    y_str = [('{:.'+str(ndigits_labels)+'f}').format(a) for a in y]
    ax.set_xticks(np.arange(0, n))
    ax.set_yticks(np.arange(0, n))
    ax.set_xticklabels(x_str, fontsize=8, rotation=45)
    ax.set_yticklabels(y_str, fontsize=8, rotation=45)
    ax.invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cb = plt.colorbar()
    if cbar_title is not None:
        cb.ax.set_title(cbar_title)


def plot_marginals(hdf5file, job_ini, refcurves=None, savedir=None, **kwargs):
    """
    Build plot of unidimensional marginal PDF or POE for each period involved in the N-D calculation

    :param hdf5file: str, path to the HDF5 containing the N-D hazard matrix results
    :param job_ini: str, path to OQ configuration file (e.g. job.ini)
    :param refcurves: dict, dictionary of reference hazard curves, used for visual comparison
                            with computed marginals. Dictionary keys should match periods read
                            in the HDF5 file,  and dictionary values are sub-dict with keys ['hdf5', 'ini']:
                            e.g. refcurves = {
                                     'SA(0.1)': {'hdf5': 'path/to/hdf5', 'ini': 'path/to/job.ini'}
                                     'PGA': {'hdf5': 'path/to/hdf5', 'ini': 'path/to/job.ini'}
                                     }
    :param savedir: str, directory path for saved figures. If None, figures are not saved (default: ./).
    :param normalize: logical, specify whether N-D PDF should be normalized (default: False)

    """
    mat, periods, logx, x = get_matrix_values_and_axes(hdf5file, job_ini)

    nd = len(mat.shape)
    if nd == 2:
        # Add special plots for the 2-D case:
        pdf, xmid = poe2pdf(mat, logx, diff_option='gradient')
        plt.show()
        plot_matrix(mat, logx[0], logx[1],
                    f'ln {periods[0]}',
                    f'ln {periods[1]}',
                    'Probability of exceedance',
                    ndigits_labels=4)
        plot_matrix(pdf, logx[0], logx[1],
                    f'ln {periods[0]})',
                    f'ln {periods[1]}',
                    'Probability Density function',
                    ndigits_labels=4)
        """
        plt.Figure()
        plt.imshow(pdf)
        plt.colorbar()
        plt.show()
        """

    marg_poe, marg_pdf = build_marginals(mat, logx, **kwargs)

    for i in range(len(periods)):
        period = str(periods[i])

        if (refcurves is not None) and (period in list(refcurves.keys())):
            ref = read_hzd_matrix(refcurves[period]['hdf5'])
            oqtmp = get_oqparam(refcurves[period]['ini'])
            ref_x = np.log(oqtmp.imtls[periods[i]])
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            # Reference 1-D hazard curve:
            ax[0].plot(ref_x, ref, label='1-D ref.')

            # Plot differences:
            # ax[1].plot(ref_x, (marg_poe[i]-ref)/ref, 'k--',
            #           marker=None, fillstyle='full', label='Rel. error')
            ax[1].plot(ref_x, marg_poe[i] - ref, 'k-',
                       marker=None, fillstyle='full', label='Abs. error')
            ax[1].set_xlabel('ln(PSA in g)')
            ax[1].grid(True)
            ax[1].legend()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

        # 1-D marginal PDF from the N-D calculation:
        xmid = logx[i][:-1]+0.5*(logx[i][1:]-logx[i][:-1])
        hc = ax[0].plot(xmid[i], marg_pdf[i], ls=':', lw=1, label='PDF')
        # 1-D marginal POE from the N-D calculation:
        ax[0].plot(xmid[i], marg_poe[i], color=hc[-1].get_color(),
                   ls='-', marker=None, fillstyle='full', label='POE')
        ax[0].set_xlabel('ln(PSA in g)')
        ax[0].set_yscale('log')
        ax[0].set_xscale('linear')
        ax[0].set_title(period)
        ax[0].legend()
        ax[0].grid(True)

        if savedir is None:
            plt.show()
        else:
            filename = savedir+os.sep+f'marginal_{period}.tif'
            save_plot(filename, h=None, size=(8, 6), keep_opened=False, tight=False)


class ProgressBar():
    """
    Progress-bar object definition
    """
    def __init__(self):
        pass

    def update(self, i: float, imax: float, title: str='', nsym: int=20):
        """ Display an ASCII progress bar with advancement level at (i/imax) %

            Parameters:
            i -- Current counter value (float)
            imax -- Maximum counter value, corresponding to 100% advancment (float)
            title -- (Optional) Title string for the progress bar
            nsym -- (Optional) Width of progress bar, in number of "=" symbols (int, default: 20)
        """
        sys.stdout.write('\r')
        fv = float(i)/float(imax)  # Fractional value, between 0 and 1
        sys.stdout.write( ('{0} [{1:'+str(nsym)+'s}] {2:3d}%').format(title, '='*ceil(fv*nsym), ceil(fv*100)) )
        if i==imax:
            sys.stdout.write('\n')
        sys.stdout.flush()

