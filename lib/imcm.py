from matplotlib import pyplot as plt
import numpy as np

class IntensityMeasureCorrelationModel():
    def __init__(self):
        self.name = None
        self.imts = (None, None)
        self.units = (None, None)
        self.bounds = [(None, None), (None, None)]

    def plot(self, xlabel=None, ylabel=None, title="Correlation model", cbar_title=None):
        """
        Produce a plot of the correlation matrix for the current model
        :param xlabel: label of X axis
        :param ylabel: label fo Y axis
        :param title: Plot title
        :param cbar_title: Colorbar title
        :return:
        """
        if xlabel is None:
            xlabel = self.imts[0]
        if ylabel is None:
            ylabel = self.imts[1]
        x = np.logspace(np.log10(self.bounds[0][0]), np.log10(self.bounds[0][1]), num=10)
        y = np.logspace(np.log10(self.bounds[1][0]), np.log10(self.bounds[1][1]), num=10)
        nx = len(x)
        ny = len(y)
        matrix = np.array((nx,ny))
        for i in range(nx):
            for j in range(ny):
                     matrix[i,j] = self.rho(x[i],y[j])
        plt.figure()
        plt.imshow(matrix, aspect='auto', interpolation="none")
        ax = plt.gca()
        x_str = [str(a) for a in x]
        y_str = [str(a) for a in y]
        ax.set_xticks(np.arange(0, nx))
        ax.set_yticks(np.arange(0, ny))
        ax.set_xticklabels(x_str, fontsize=8)
        ax.set_yticklabels(y_str, fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        cb = plt.colorbar()
        if cbar_title is not None:
            cb.ax.set_title(cbar_title)

    def rho(self, value1, value2):
        pass


class BakerCornell2006(IntensityMeasureCorrelationModel):
    """
    Baker & cornell (2006) inter-spectral correlation model.
    Records Database: PEER Strong Motion Database (2000)
    Related GMPEs:
            Abrahamson and Silva, 1997;
            Boore et al., 1997;
            Campbell, 1997
    """
    def __init__(self):
        IntensityMeasureCorrelationModel.__init__(self)
        self.name = "BakerCornell2006"
        self.imts = ('SA', 'SA')
        self.units = ('s.', 's.')  # Periods in seconds
        self.bounds = ((0.05, 5), (0.05, 5))

    def rho(self, T1, T2):
        """
        Returns the correlation coefficient for ground-motions observed at two
        different periods T1 and T2, based on the Baker & Cornell (2006) study
        """
        Tmin = min(T1, T2)
        Tmax = max(T1, T2)
        if (Tmin < 0.05) or (Tmax > 5):
            raise ValueError('Periods are beyond the uppper/lower bounds of the {} model'.format(self.name))
        if Tmin < 0.189:
            r = 1 - np.cos(np.pi / 2 - (0.359 + 0.163 * np.log(Tmin / 0.189)) \
                           * np.log(Tmax / Tmin))
        else:
            r = 1 - np.cos(np.pi / 2 - 0.359 * np.log(Tmax / Tmin))
        return r


class BakerJayaram2008(IntensityMeasureCorrelationModel):
    """
    Baker & Jayaram (2008) inter-spectral correlation model.
    Records Database: NGA (Next Generation Attenuation project)
    Related GMPEs:
            Abrahamson and Silva, 2008;
            Boore and Atkinson, 2008;
            Campbell and Bozorgnia, 2008;
            Chiou and Youngs, 2008
    """
    def __init__(self):
        IntensityMeasureCorrelationModel.__init__(self)
        self.name = "BakerJayaram2008"
        self.imts = ('SA', 'SA')
        self.units = ('s.', 's.')  # Periods in seconds
        self.bounds = ((0.01, 10), (0.01, 10))

    def rho(self, T1, T2):
        """
        Returns the correlation coefficient for ground-motions observed at two
        different periods T1 and T2, based on the Baker & Jayaram (2008) study.
        """
        Tmin = min(T1, T2)
        Tmax = max(T1, T2)
        if (Tmin < 0.01) or (Tmax > 10):
            raise ValueError('Periods are beyond the uppper/lower bounds of the {} model'.format(self.name))
        else :
            C1=1-np.cos(np.pi/2 -0.366*np.log(Tmax/max(Tmin,0.109)))
            if Tmax<0.2:
                C2=1-0.105*(1- 1/(1+np.exp(100*Tmax-5))) * ((Tmax-Tmin)/(Tmax-0.0099))
            else:
                C2=0
            if Tmax<0.109:
                C3=C2
            else:
                C3=C1
            C4=C1 + 0.5*(np.sqrt(C3)-C3)*(1+np.cos(np.pi*Tmin/0.109))
            if Tmax<0.109:
                return C2
            elif Tmin>0.109:
                return C1
            elif Tmax<0.2:
                return min(C2,C4)
            else :
                return C4
