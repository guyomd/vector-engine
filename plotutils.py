import sys
from math import ceil
from matplotlib import pyplot as plt
import numpy as np

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

