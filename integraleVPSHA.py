import numpy as np
from scipy.stats import multivariate_normal, norm, mvn
import gsim

#import UHS_ConditionalReturnPeriod

### fct dans UHS_ConditionalReturnPeriod :
#
# plot_matrix(M, x, y, xlabel, ylabel, title, cbar_title=None)  -> histogramme
# arrays2dict(periods, sa, pga_period=0)                        -> dictionnaire [dict], Périodes [liste], SA [liste]
# means_and_cov_matrix(period_keys, gmpe, corr, scenario)       -> lnSa [?], Covariance [?]
# BakerCornell2006(T1, T2)                                      -> coef correlation [1 valeur]

################################################################################



def means_and_cov_matrix(period_keys, gmpe, corr, scenario):
    """
    Build the symmetrical covariance matrix for the multivariate (correlated)
    ground-motion model
    corr: Correlation matrix
    """
    def is_pos_semidef(x):
        return np.all(np.linalg.eigvals(x) >= 0)

    n = len(period_keys)
    per = [ float(s[3:-1]) if s.startswith('SA(') else 0 for s in period_keys ]
    gmpe.means_and_stddevs(period_keys,
                           scenario.m,
                           scenario.r,
                           scenario.vs30,
                           scenario.rake)
    lnSA = np.squeeze(gmpe.mean_ln)
    lnSTD = np.squeeze(gmpe.sigma_ln)
    if not is_pos_semidef(corr):
        print(f'ERROR: Correlation matrix is not positive, semi-definite')
        val = np.linalg.eigvals(corr)
        print(f'EIGENVALUES: {val}')
        print(f'ADVICE: REPLACE negative eigenvalues with 0')
        exit(2)
    # Build covariance matrix:
    D = np.diag(lnSTD)
    C = D@corr@D
    return lnSA, C

# Ground-motion correlation model:
def BakerCornell2006(T1, T2):
    """
    Return the correlation coefficient for ground-motions observed at two
    different periods T1 and T2, based on the Baker & Cornell (2006) study
    """
    Tmin = min(T1, T2)
    Tmax = max(T1, T2)
    if (Tmin<0.05) or (Tmax>5):
        raise ValueError('Periods are beyond the uppper/lower bounds of this model')
    if Tmin<0.189:
        r = 1 - np.cos(np.pi/2 - (0.359+0.163*np.log(Tmin/0.189)) \
                       *np.log(Tmax/Tmin))
    else:
        r = 1 - np.cos(np.pi/2 - 0.359*np.log(Tmax/Tmin))
    return r


class fM:
    def GR (b,m,Mmin):
       return b*np.log(10)*10**(-b*(m-Mmin))
    def GR_tr (b,m,Mmin,Mmax):
       return (b*np.log(10)*10**(-b*(m-Mmin))) / (1-10**(-b*(Mmax-Mmin)))


##################################################################################

# entrées #################
vs30 = 800
gmpe = gsim.OQgsim('AmeriEtAl2017Repi')  # GMPE

per = np.array([ 2, 1, 0.5, 0.333, 0.25, 0.2, 0.15, 0.1, 0.0667, 0.05, 0.033, 0.025, 0.02, 0.01, 0])
per = np.array([0, 1])  # the length of (per) give the dimension of the cdf "phi_chap"!!

SA = [0.001, 0.01, 0.1, 1.0]  # target accelerations for each dimension

na = len(SA)
rho = BakerCornell2006  # Pointer to correlation model


point_sources = []


Dm=[]  # liste : 1 nb par source



######################################################
MRE = np.zeros((na,len(per))) # on veut np.zeros((4,4)) dans le cas 2D, mais on veut que ce soit extensible a N-D... 
"""fct np.zeros ne marche que pour 1,2 ou 3 dimensions, trouver une autre méthode pour dimension > 3"""
# Build pseudo-correlation matrix:
nper = len(per)
CORR = np.zeros((nper,nper))
for i in range(nper):
    for j in range(i,nper):
        CORR[i, j] = rho(per[i], per[j])
        if i != j:
            CORR[j,i] = CORR[i,j]



n=len(point_sources)



for i1 in range(na):  # 1ere periode
    for i2 in range(na): # 2eme periode
        # Comment generaliser le nombre de periode utilisées sans fixer a priori le nombre de boucles for ?
        for i in range (n):  # Loop on point-sources:  "pseudo integrale distance"
            # Parameters a recuperer a partir de l'objet point source ou area-source:
            Mmin = 5.0
            Mmax = 7.0
            b = todo
            nu = todo
            jj=(Mmax-Mmin)/Dm
            for j in range (jj):    # integrale magnitude
                m = Mmin + Dm / 2 + j * Dm
                sce.m = m
                sce.r = todo
                sce.vs30 = vs30
                sce.rake = todo
                # pour periode de 1 s., utiliser 'SA(1.0)'. Les valeurs possibles sont 'PGV', 'PGA' (T=0) et 'SA(x)' où x est la valeur de la periode
                # per_keys est une liste de ces valeurs
                means, C = means_and_cov_matrix(per_keys, gmpe, CORR, sce)
                # means is the ln(SA) in units of g
                # C is the std. dev. of ln(SA)
                print_info(f'INFO: COV has shape {C.shape}:\n{C}')
                print_info(f'INFO: MEANS has shape {means.shape}:\n{means}')


                # Compute return period:
                lower = np.array([SA[i1], SA[i2]])
                upper = np.inf * np.ones_like(lower)
                abseps = 0.001
                ndigits = int(-round(np.log10(abseps)))
                phi_chap, error_code = mvn.mvnun(lower, upper, means, C, abseps=abseps, maxpts=100000)  # optimal :  abseps=1e-6, maxpts=len(lower)*1000

                

                fm = fM.GR_tr (b,m,Mmin,Mmax) # ou fM.GR(b,m,Mmin) 
                MRE[i1,i2] = MRE[i1,i2] + nu*phi_chap*fm*Dm #*Dr # refrechir au cas de Dr (necessaire?)





