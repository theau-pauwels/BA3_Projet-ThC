import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as sc

def draw_repfreq(num, den, w_min, w_max, title = '', savename = ''):

    '''
    Dessine la réponse en fréquence du filtre en fonction des polynômes de H(p) = Num(p)/Den(p) entre
    les valeurs 10^w_min et 10^w_max. Ainsi, si w_min = 0 et w_max = 1, la courbe de Bode sera dressée
    entre 10^0 et 10^1 en échelle logarithmique

    Example : H(p) = (p+1)/(p+2) 
    inputs : num=[1,1], den=[1,2] (<= facteurs multiplicatifs du polynôme), w_min = 0, w_max = 1
    outputs : None
    '''

    plt.figure()
    num = np.array(num) 
    den = np.array(den)

    wIn = np.logspace(w_min, w_max, 10000000)
    wOut, hOut = sc.freqs(num, den, wIn)
    
    plt.semilogx(wOut, 20*np.log10(np.abs(hOut)))
    
    plt.grid()
    
    ### Ajout du titre et sauvegarde
    if (savename != '') & (title != ''):
        plt.title(title)
        plt.savefig(savename)
    ###
    
    plt.show()

def compute_roots(poly):

    '''
    Calcule la racine d'un polynôme 

    Example : F(p) = (p+1) 
    inputs : num=[1,1]
    outputs : racine du polynôme
    '''
    return np.roots(np.array(poly))