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

class TraversaBremaud2008(IntensityMeasureCorrelationModel):
    """
    Traversa & Bremaud (2020, unpublished) inter-spectral correlation model.
    Records Database: RESORCE, 2019 edition
    Related GMPEs:
            Traversa et al, 2020
    """
    def __init__(self):
        IntensityMeasureCorrelationModel.__init__(self)
        self.name = "Europe"
        self.imts = ('SA', 'SA')
        self.units = ('s.', 's.')  # Periods in seconds
        self.bounds = ((0.04, 7), (0.04, 7))
        self.liste=self.CreerListe()
        self.Tableau=self.CreerTableau()


    def CreerListe(self):
        liste=dict()
        liste[0.04]=1;liste[0.05]=2;liste[0.075]=3;liste[0.1]=4
        liste[0.11]=5;liste[0.12]=6;liste[0.13]=7;liste[0.14]=8
        liste[0.15]=9;liste[0.16]=10;liste[0.17]=11;liste[0.18]=12
        liste[0.19]=13;liste[0.2]=14;liste[0.22]=15;liste[0.24]=16
        liste[0.26]=17;liste[0.28]=18;liste[0.3]=19;liste[0.32]=20
        liste[0.34]=21;liste[0.36]=22;liste[0.38]=23;liste[0.4]=24
        liste[0.42]=25;liste[0.44]=26;liste[0.46]=27;liste[0.48]=28
        liste[0.5]=29;liste[0.55]=30;liste[0.6]=31;liste[0.65]=32
        liste[0.7]=33;liste[0.75]=34;liste[0.8]=35;liste[0.85]=36
        liste[0.9]=37;liste[0.95]=38;liste[1]=39;liste[1.1]=40
        liste[1.2]=41;liste[1.3]=42;liste[1.4]=43;liste[1.5]=44
        liste[1.6]=45;liste[1.7]=46;liste[1.8]=47;liste[1.9]=48
        liste[2]=49;liste[2.2]=50;liste[2.4]=51;liste[2.6]=52
        liste[2.8]=53;liste[3]=54;liste[3.2]=55;liste[3.4]=56
        liste[3.6]=57;liste[3.8]=58;liste[4]=59;liste[4.2]=60
        liste[4.4]=61;liste[4.6]=62;liste[4.8]=63;liste[5]=64
        liste[5.5]=65;liste[6]=66;liste[6.5]=67;liste[7]=68
        return(liste)


    def CreerTableau(self):
        import csv
        fname="data/TraversaBremaud2020_all_coeff_corr_SA.csv"
        Tableau=[]
        with open(fname, newline='')as f:
            reader = csv.reader(f)
            for row in reader:
                Tableau.append(row)
        for i in range (len(Tableau)):
            for j in range (len(Tableau[0])):
                if (i,j)==(0,0):
                    pass
                else:
                    Tableau[i][j]=float(Tableau[i][j])
        return Tableau

    def rho(self, T1, T2):
        if (T1 not in self.liste) or (T2 not in self.liste):
            raise ValueError('Periods are not in the {} model'.format(self.name))
        else:
            r=self.Tableau[self.liste[T1]][self.liste[T2]]
        return r
