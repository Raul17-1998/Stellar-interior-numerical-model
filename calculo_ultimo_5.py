from calculo_errores_final_5 import *

radio_final, lum_final, temp_final = calculo_error(M_tot,X,Y,R_tot,L_tot,T_central)

print("           ")

print("----------------------------------------------------------------------")
print("--------------· VALORES FINALES CON LOS QUE SE TRABAJA ·--------------")
print("----------------------------------------------------------------------")

print("Masa total = ", M_tot)
print("X = ", X)
print("Y = ", Y)
print("Radio total = ", round(radio_final,3))
print("Luminosidad total = ", lum_final)
print("Temperatura central = %.3f" % temp_final)
print("   ")

# =============================================================================
# ARCHIVO 3/3
# 
# Este archivo calcula otros datos a partir de los resultados. Aparte de una
# gráfica de las distintas magnitudes en función del radio, también devuelve
# un diagrama HR en el que se sitúa la estrella modelo final representada.
# =============================================================================

funcion_graficas(M_tot,X,Y,radio_final,lum_final,temp_final)

print("A partir de estos datos...")
sigma = 5.6704e-5
L_sol = 3.85*1e33
R_sol = 6.96*1e10
Teff_sol = 5780.0
L_tot = lum_final*1e33
R_tot = radio_final*1e10
Lu = L_tot/L_sol
Ra = R_tot/R_sol

# Flujo
Teff = (L_tot/(4*np.pi*R_tot**2*sigma))**0.25
print(" La temperatura eficaz es ", Teff, " K")
# Ley de Wien
long_max = 0.002898/Teff
print(" La longitud de onda máxima de emisión de la estrella es ", long_max, " m")
# Ley de Pogson -- Bolométricas
mag_abs_sol = 4.74
mag_abs = (4.74 - 2.5*np.log10(Lu))*mag_abs_sol
print(" La magnitud bolométrica absoluta de la estrella es: ", mag_abs)


# =============================================================================
# DIAGRAMA HR
# =============================================================================

import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import math
import pandas as pd

rotated_labels = []
def text_slope_match_line(text, x, y, line):
    global rotated_labels

    # pendiente
    xdata, ydata = line.get_data()

    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    rotated_labels.append({"text":text, "line":line, "p1":np.array((x1, y1)), "p2":np.array((x2, y2))})

def update_text_slopes(ax):
    global rotated_labels

    for label in rotated_labels:
        # Transformamos los datos de las pendientes
        text, line = label["text"], label["line"]
        p1, p2 = label["p1"], label["p2"]

        ax = ax

        sp1 = ax.transData.transform_point(p1)
        sp2 = ax.transData.transform_point(p2)

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        slope_degrees = math.degrees(math.atan(rise/run))

        text.set_rotation(slope_degrees)
        
        
# Importamos datos de la secuencia principal
import requests
url = 'https://en.wikipedia.org/wiki/Main_sequence'
html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[1]
df.to_csv('main_seq_data.csv')

print("\n La secuencia principal contiene estrellas tipo como las de la siguiente lista \n", df)

main_seq_lumis = df['Luminosity, L/L☉']
main_seq_teffs = df['Temp. (K)']

# Fondo negro y textos en blanco
plt.style.use('dark_background')
fig = plt.figure(figsize=(10,9))
ax = plt.gca()
line, = ax.plot([],[], '--', lw=2.5, color='k')
ax.set_xlim(1000,40000)
ax.set_ylim(10**-5,10**6)
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_xlabel(r'$Temperatura\ efectiva\ (K)$', fontsize=15, c='white')
ax.set_ylabel(r'$Luminosidad\ (L/L_{\odot})$', fontsize=15, c='white')
ax.set_title('Diagrama HR', fontsize=20, c='white')
ax.grid(False)

# Construimos vectores para radio y temperatura para los radios constantes
R = np.array([0.001, 0.01, 0.1, 1., 10., 100., 1000.])
T = np.linspace(1000., 50000., num=50)

# Posición de las anotaciones de los radios constantes
xloc = np.array([38000, 38000, 38000, 38000, 28000, 15000, 4750])
yloc = (xloc/Teff_sol)**4*R**2

# Dibujamos líneas de radio constante
for i in range(len(R)):
    L = pow(T/Teff_sol, 4)*pow(R[i], 2)
    greyline, = ax.plot(T, L, '--', c='gray', lw=0.8, zorder=1)
    t = ax.annotate(str(R[i])+' $R_{\odot}$ ', xy=(xloc[i], yloc[i]), xytext=(12, 0),
                    textcoords='offset points', horizontalalignment='left', verticalalignment='center_baseline',
                    color='grey')
    text_slope_match_line(t, 40000, L[-1], greyline)
    
update_text_slopes(ax)

vals = main_seq_teffs
norm = plt.Normalize(vals.min()-2200, vals.max()-37500) # Variar en función de lo que queramos
colours = plt.cm.RdYlBu(norm(vals))

sizes = df['Radius, R/R☉']

plt.scatter(main_seq_teffs, main_seq_lumis, s=sizes*200, c=colours)
ax.annotate('Sol', xy=(5250, 1), color='Yellow')
plt.scatter(Teff_sol, 1, s=sizes[9]*200, facecolors='none', marker='o', edgecolors='k')
plt.scatter(Teff, Lu, s=Ra*200, facecolors='none', marker='o', edgecolors='white')
plt.scatter(Teff, Lu, s=Ra*100, c='white', marker='x', label='Estrella modelo')
plt.legend()

Mm = np.zeros(len(main_seq_lumis))
for i in range(len(Mm)):
    Mm[i] = (4.74 - 2.5*np.log10(main_seq_lumis[i]))

ax2 = ax.twinx()
ax2.set_ylim(-10,17)
ax2.invert_yaxis()
ax2.set_ylabel('Magnitud bolométrica absoluta', rotation=270, fontsize=15, c='white')
ax2.scatter(main_seq_teffs, Mm, facecolor='white', marker='o', edgecolors='none')
ax2.tick_params(axis ='y', color='white')
