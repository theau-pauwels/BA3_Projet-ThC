# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:37:57 2024

@author:    Pauwels Theau
            Rogge Mathys
"""
from utils import *
from zplane import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc

#%%
"""Partie 1.1"""

# paramètres
f_tri = 1                           # fréquence de l'onde triangulaire à 1 [Hz]
A_tri = 1.2                         # amplitude de l'onde triangulaire à 1.2 [V]
f_sin = 1e-1                        # fréquence du signal sinusoïdal à 0.1 [Hz]
A_sin = 1                           # amplitude du signal sinusoïdal à 1 [V]
f_e = 1e3                           # fréquence d'échantillonage à 1 [kHz]
T = 10                              # durée d'échantillonnage à 10 [s]
t = np.arange(0, T, 1/f_e)          # temps pour les échantillons

# génération des fcts
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 1 [Hz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 0.1 [Hz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM

# Point 1.1.1
# affichage des fcts de base
plt.subplot(2, 1, 1)
plt.plot(t, sin_wave,color='b')
plt.plot(t, tri_wave,color='r')
plt.title('Signaux pour la PWM')
plt.ylabel('Amplitude')

# affichage de la PWM
plt.subplot(2, 1, 2)
plt.plot(t, pwm_signal, label='Signal PWM', color='g')
plt.title('Modulation PWM')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# affichage et enregistrement des fcts et de la PWM
plt.savefig("1.1.1 - PWM btw sin and tri.png")
plt.show()

#%%

# paramètres
f_tri = 20e3                        # fréquence de l'onde triangulaire à 20 [kHz]
A_tri = 1.2                         # amplitude de l'onde triangulaire
f_sin = 1e3                         # fréquence du signal sinusoïdal à 1 [kHz]
A_sin = 1                           # amplitude du signal sinusoïdal
f_e = 1e6                           # fréquence d'échantillonnage 1 [MHz]
T = 5e-3                            # durée d'échantillonnage 5 [ms]
t = np.arange(0, T, 1/f_e)          # temps pour les échantillons

# génération des fcts
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM

# Point 1.1.4
# affichage des fcts de base
#plt.subplot(2, 1, 1)
plt.plot(t, tri_wave, color='b')
plt.plot(t, sin_wave, label="Onde triangulaire (20 [kHz]) et signal sinusoïdal (1 [kHz])", color='r')
plt.title("Onde triangulaire (20 [kHz]) et signal sinusoïdal (1 [kHz])")
plt.grid()
plt.savefig("1.1.4 - sin and tri.png")
plt.show()

# affichage de la PWM
#plt.subplot(2, 1, 2)
plt.plot(t, pwm_signal, label="Signal PWM", color='g')
plt.title("Signal PWM modulé")
plt.grid()

# affichage et enregistrement des fcts et de la PWM
plt.savefig("1.1.4 - PWM btw sin and tri.png")
plt.show()

#%%

#Point 1.1.5
# paramètres
f_tri = 20e3                        # fréquence de l'onde triangulaire à 20 [kHz]
A_tri = 1.2                         # amplitude de l'onde triangulaire
f_sin = 1e3                         # fréquence du signal sinusoïdal à 1 [kHz]
A_sin = 1                           # amplitude du signal sinusoïdal
f_e = 1e6                           # fréquence d'échantillonnage 1 [MHz]
T = 5e-3                            # durée d'échantillonnage 5 [ms]

### f_e/2
# génération des fcts
t = np.arange(0, T, 2/f_e)                                      # temps pour les échantillons
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM
# tracé
plt.subplot(3, 1, 1)
plt.plot(t, pwm_signal, color='g', label='f_e/2')
plt.title("Signal PWM modulé")
plt.ylabel('Amplitude [V]')
plt.grid(True)
plt.legend()

### f_e/3
# génération des fcts
t = np.arange(0, T, 3/f_e)                                      # temps pour les échantillons
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM
# tracé
plt.subplot(3, 1, 2)
plt.plot(t, pwm_signal, color='g', label='f_e/3')
plt.ylabel('Amplitude [V]')
plt.grid(True)
plt.legend()

### f_e/10
# génération des fcts
t = np.arange(0, T, 10/f_e)                                     # temps pour les échantillons
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM
# tracé
plt.subplot(3, 1, 3)
plt.plot(t, pwm_signal, color='g', label='f_e/10')
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude [V]')
plt.grid(True)
plt.legend()

# affichage et enregistrement des différentes f_e
plt.savefig("1.1.5 - PWM btw sin and tri.png") # Sauvegarde au format PNG
plt.show()

#%%
"""Partie 1.3"""

#Point 1.3.1
# paramètres
f_tri = 20e3                        # fréquence de l'onde triangulaire à 20 [kHz]
A_tri = 1.2                         # amplitude de l'onde triangulaire
f_sin = 1e3                         # fréquence du signal sinusoïdal à 1 [kHz]
A_sin = 1                           # amplitude du signal sinusoïdal
f_e = 1e6                           # fréquence d'échantillonnage 1 [MHz]
T = 5e-3                            # durée d'échantillonnage 5 [ms]
t = np.arange(0, T, 1/f_e)          # temps pour les échantillons

# génération des fcts
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM
pwm_signal *= 100                                               # amplifié 100 fois

# affichage de la PWM
#plt.subplot(2, 1, 2)
plt.plot(t, pwm_signal, label='Signal PWM', color='b')
plt.title('Modulation PWM')
plt.xlabel('Temps [ms]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# affichage et enregistrement des fcts et de la PWM
plt.savefig("1.3.1 - ampl_PWM btw cos and tri.png")
plt.show()

#%%

# paramètres
f_tri = 20e3                        # fréquence de l'onde triangulaire à 20 [kHz]
A_tri = 1.2                         # amplitude de l'onde triangulaire
f_sin = 1e3                         # fréquence du signal sinusoïdal à 1 [kHz]
A_sin = 1                           # amplitude du signal sinusoïdal
f_e = 1e6                           # fréquence d'échantillonnage 1 [MHz]
T = 5e-3                            # durée d'échantillonnage 5 [ms]
t = np.arange(0, T, 1/f_e)          # temps pour les échantillons

# génération des fcts
tri_wave = sc.sawtooth(2 * np.pi * f_tri * t, 0.5) * A_tri      # signal triangle à 20 [kHz]
sin_wave = np.sin(2 * np.pi * f_sin * t) * A_sin                # sinus à 1 [kHz]
pwm_signal = (tri_wave < sin_wave).astype(float)                # signal PWM
pwm_signal *= 100                                               # amplifié 100 fois

# calcul de la transformée de Fourier
x, y = sc.freqz(pwm_signal, 1, 2**(13))

# affichage et enregistrement des fcts et de la PWM autour de 1 [kHz]
plt.plot(x * f_e / (2 * np.pi), 20*np.log(np.abs(y)))
plt.title('Transformée de Fourier du signal PWM')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.savefig("1.3.2 - Fourier transform of ampl_PWM.png")
plt.show()

# affichage et enregistrement des fcts et de la PWM autour de 1 [kHz]
plt.plot(x * f_e / (2 * np.pi), 20*np.log(np.abs(y)))
plt.xlim(0, 2000)                                               # centrage sur 1 [kHz]
plt.title('Transformée de Fourier du signal PWM centrée en 1 [kHz]')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.savefig("1.3.2 - Fourier transform of ampl_PWM centered on 1000 Hz.png")
plt.show()

# affichage et enregistrement des fcts et de la PWM autour de 20 [kHz]
plt.plot(x * f_e / (2 * np.pi), 20*np.log(np.abs(y)))
plt.xlim(19000, 21000)                                          # centrage sur 20 [kHz]
plt.title('Transformée de Fourier du signal PWM centrée en 20 [kHz]')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.savefig("1.3.2 - Fourier transform of ampl_PWM centered on 20 000 Hz.png")
plt.show()

#%%
'''Partie 2'''

# paramètres
f_pass = 16e3                           # fréquence passante en [Hz]
f_stopband = 19e3                       # fréquence bloquante en [Hz]
A_pass = 1.53                           # amplitude atténuation passante en [dB]
A_stopband = 40                         # amplitude atténuation bloquante en [dB]
w_pass = 2 * np.pi * f_pass             # pulsation passante en [rad/s]
w_stopband = 2 * np.pi * f_stopband     # pulsation bloquante en [rad/s]
W_pass = 1                              # fréquence théorique de cassure
W_stopband = np.pi*2*f_stopband/w_pass  # fréquence réelle

#%% Butterworth

# calcul des coeffs de Butterworth
N, w_b = sc.buttord(W_pass, W_stopband, A_pass, A_stopband, analog=True)
f_b = w_b/(2*np.pi)

# affichage des coeffs de Butterworth
print("Ordre du filtre : ", N)
print("Fréquence de coupure : ", f_b)

# affichage et enregistrement des zéros et de la fct de transfert de Butterworth
b, a = sc.butter(N, w_b, btype='low', output='ba', analog=True)
zplane(b, a, "2.0.0 - zplane de Butterworth.png", "zplane de Butterworth")
print(b, a)

# affichage et enregistrement de la réponse en fréquence
w, h = sc.freqs(b,a)
plt.semilogx(w* f_e / (2 * np.pi), 20 * np.log10(abs(h)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.title('Réponse en fréquence Butterworth')
plt.grid(True)
plt.savefig("2.0.0 - Réponse en fréquence Butterworth.png")
plt.show()

#%% Chebychev 1

# calcul des coeffs de Chebychev 1
N,w_b = sc.cheb1ord(W_pass, W_stopband, A_pass, A_stopband, analog=True)
f_b = w_b/(2*np.pi)

# affichage des coeffs de Chebychev 1
print("Ordre du filtre : ", N)
print("Fréquence de coupure : ", f_b)

# affichage et enregistrement des zéros et de la fct de transfert de Chebychev 1
b,a = sc.cheby1(N, A_pass, w_b, analog=True)
zplane(b, a, "2.0.0 - zplane de Chebychev 1.png", "zplane de Chebychev 1")
print(b,a)

# affichage et enregistrement de la réponse en fréquence
w, h = sc.freqs(b, a)
plt.semilogx(w* f_e / (2 * np.pi), 20 * np.log10(abs(h)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.title('Réponse en fréquence Chebychev 1')
plt.grid(True)
plt.savefig("2.0.0 - Réponse en fréquence Chebychev 1.png")
plt.show()

#%% Cauer

# calcul des coeffs de Chebychev 1
N, w_b = sc.ellipord(W_pass, W_stopband, A_pass, A_stopband, analog=True)
f_b = w_b/(2*np.pi)

# affichage des coeffs de Cauer
print("Ordre du filtre : ", N)
print("Fréquence de coupure : ", f_b)


# affichage et enregistrement des zéros et de la fct de transfert de Cauer
b, a = sc.ellip(N, 1, A_stopband, w_b, analog=True)
zplane(b, a, "2.0.0 - zplane de Cauer.png", "zplane de Cauer")
print(b,a)

# affichage et enregistrement de la réponse en fréquence
w, h = sc.freqs(b,a)
plt.semilogx(w* f_e / (2 * np.pi), 20 * np.log10(abs(h)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.title('Réponse en fréquence Cauer')
plt.grid(True)
plt.savefig("2.0.0 - Réponse en fréquence Cauer.png")
plt.show()

#%% Synthèse en cascade RCAO

# paramètres
w_min = -2
w_max = 2

# calcul des zeros pour les fcts des cellules
num = np.roots(b)
den = np.roots(a)

# calcul des diffs num et den
num1 = np.poly([num[0], num[1]])
den1 = np.poly([den[0], den[1]])
num2 = np.poly([num[2], num[3]])
den2 = np.poly([den[2], den[3]])
den3 = np.poly([den[4]])

# affichage des diffs H(p)
print("num1 = ", num1)
print("den1 = ", den1, '\n')
print("num2 = ", num2)
print("den2 = ", den2, '\n')
print("den3 = ", den3, '\n')

# gains K_i des filtres
# calcul et affichage du gain K_1
K_1 = np.polyval(np.poly([num[0], num[1]]), 0)/ np.polyval(np.poly([den[0], den[1]]), 0)
print("Gain K_1 =", 20*np.log10(abs(K_1)), "[dB] = ", K_1)
print("rho_1 pôles =", (den1[-1]**2), '\n')
# calcul et affichage du gain K_2
K_2 = np.polyval(np.poly([num[2], num[3]]), 0)/ np.polyval(np.poly([den[2], den[3]]), 0)
print("Gain K_2 =", 20*np.log10(abs(K_2)), "[dB] = ", K_2)
print("rho_2 pôles =", den2[-1]**2, '\n')
# calcul et affichage du gain K_3
K_3 = 1 / np.polyval(np.poly([den[4]]), 0)
print("Gain K_3 =", 20*np.log10(abs(K_3)), "[dB] = ", K_3, '\n')

# affichage et enregistrement des rep freq des H(p)
# filtre 1
title = 'Filtre 1 : num = p² + 3.11, deno = (p² + 0.10p + 1)*K_1'
savename = '2.0.4 - ' + title[0:8] + '.png'
draw_repfreq(num1, den1*K_1, w_min, w_max, title, savename)
# filtre 2
title = 'Filtre 2 : num = p² + 1.57, deno = (p² + 0.44p + 0.6)*K_2'
savename = '2.0.4 - ' + title[0:8] + '.png'
draw_repfreq(num2, den2*K_2, w_min, w_max, title, savename)
# filtre 3
title = 'Filtre 3 : num = 1, deno = (p + 0.39)*K_3'
savename = '2.0.4 - ' + title[0:8] + '.png'
draw_repfreq([1], den3*K_3, w_min, w_max, title, savename)

#%% affichage et enregistrement de la rep freq de H(p)
# multiplication des polynômes avec np.polymul
num_total = np.polymul(num1, num2)                                  # Produit des numérateurs
den_total = np.polymul(np.polymul(den1, den2), den3)*K_1*K_2*K_3    # Produit des dénominateurs
title = 'Filtre de Cauer : H(p) = H_1(p)*H_2(p)*H_3(p)'
savename = '2.0.4 - ' + title[10:15] + '.png'
draw_repfreq(num_total, den_total, w_min, w_max, title, savename)
