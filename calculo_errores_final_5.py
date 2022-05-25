# =============================================================================
# ARCHIVO 2/3
# 
# Este archivo contiene la búsqueda del error relativo mínimo para los valores
# dados. Devuelve también mapas de colores para poder visualizar dónde está
# el error mínimo.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import time

from calculo_estrella_final_5 import *
from pandas import *

print("           ")

print("----------------------------------------------------------------------")
print("---· BUSCANDO VALORES ÓPTIMOS PARA EL RADIO Y TEMPERATURA CENTRAL ·---")
print("----------------------------------------------------------------------")

# Si hay problemas, copiar y pegar el código que hay fuera de "funcion"
# de calculo_estrella

def calculo_error(M_tot,X,Y,R_tot,L_tot,T_central):
    decimales = [0,1,2]
    # decimales & m están definidos para que salga los decimales exactos
    
    for m in decimales:
        
        # En caso de querer ajustar un intervalo de manera manual...
        # if m == 0:
            # radio_ajuste = np.linspace(45,48,11) #1*10**-m
            # lum_ajuste = np.linspace(490,500,11)
        # elif m==1 or m==2:
            # lum_ajuste = np.linspace(L_tot-10*10**-m,L_tot+10*10**-m,11) #10*10**-m
        # elif m == 1 or m == 2:
            # radio_ajuste = np.linspace(R_tot-2.5*10**-m,R_tot+2.5*10**-m,11) #1*10**-m
            # lum_ajuste = np.linspace(L_tot-10*10**-m,L_tot+10*10**-m,11)
        
        radio_ajuste = np.linspace(R_tot-2.5*10**-m,R_tot+2.5*10**-m,11) #1*10**-m
        lum_ajuste = np.linspace(L_tot-10*10**-m,L_tot+10*10**-m,11)
    
        rrll = np.zeros((np.size(lum_ajuste),np.size(radio_ajuste)))
        TT_ajuste = np.zeros((np.size(lum_ajuste),np.size(radio_ajuste)))
        
        print("----------------------------------------------------------------------")
        print("Intervalo de radios: ", radio_ajuste)
        print("Intervalo de luminosidades: ", lum_ajuste)
        print("   ")
        
        start = time.time()
        for t in range(len(lum_ajuste)):
            print("Trabajando con L:", lum_ajuste[t])
            for q in range(len(radio_ajuste)):
                # print(" · Trabajando con R:", radio_ajuste[q])
                l_ajuste = lum_ajuste[t]
                r_ajuste = radio_ajuste[q]
                rrll[t,q], TT_ajuste[t,q] = funcion(M_tot,X,Y,r_ajuste,l_ajuste,T_central)            
        # print(tabulate(rrll, tablefmt='fancy_grid', stralign='center', floatfmt='.4f'))
        # print(tabulate(TT_ajuste, tablefmt='fancy_grid', stralign='center', floatfmt='.4f'))
        np.savetxt("rrll_values.txt",rrll)
        np.savetxt("TT_central_values.txt",TT_ajuste)
        end = time.time()
        
        print("   ")
        print("Tiempo: ", int((end-start)/60), "minutos y", ((end-start)/60-int((end-start)/60))*60, "segundos")
        
        
        minimo_err = np.min(rrll) #mínimo de la matriz
        
        print("   ")
        print("El valor mínimo para el error relativo es", minimo_err)
        
        ij = np.unravel_index(np.argmin(rrll),rrll.shape)
        R_tot = radio_ajuste[ij[1]]
        L_tot = lum_ajuste[ij[0]]
        T_central = TT_ajuste[ij[0],ij[1]]
        
        print("El mejor valor para el radio es", R_tot)
        print("El mejor valor para la luminosidad es", L_tot)
        print("El mejor valor para la temperatura es", T_central)
        print("----------------------------------------------------------------------")
        print("   ")
       
        
        # Tablas y figuras
        
        df = DataFrame(rrll, index=np.round(lum_ajuste,2), columns=np.round(radio_ajuste,2))
        df_T = DataFrame(TT_ajuste, index=np.round(lum_ajuste,2), columns=np.round(radio_ajuste,2))
        
        sns.set(font_scale=1.5)
        
        fig = plt.figure(num=None, figsize=(10, 8))
        res = sns.heatmap(df, annot=False, robust=True, cmap='gist_earth_r', cbar_kws = dict(use_gridspec=False,location="top"))
        plt.title("Error mínimo (%)", fontsize=22)
        plt.xlabel("$R_{tot}$", fontsize=22)
        plt.ylabel("$L_{tot}$", fontsize=22)
        plt.show()
        
        fig = plt.figure(num=None, figsize=(20, 10))
        # Plot 1:
        plt.subplot(1, 2, 1)
        res = sns.heatmap(df, annot=False, robust=True, cmap='gist_earth_r', cbar_kws = dict(use_gridspec=False,location="top"))
        plt.title("Error mínimo (%)", fontsize=25)
        plt.xlabel("$R_{tot}$", fontsize=22)
        plt.ylabel("$L_{tot}$", fontsize=22)
        # Plot 2:
        plt.subplot(1, 2, 2)
        res = sns.heatmap(df_T, annot=False, robust=True, cmap='gist_earth_r', cbar_kws = dict(use_gridspec=False,location="top"))
        plt.title("Temperatura central óptima", fontsize=25)
        plt.xlabel("$R_{tot}$", fontsize=22)
        plt.ylabel("$L_{tot}$", fontsize=22)
        
        plt.suptitle("$X=$"+str(X)+", $Y=$"+str(Y)+", $M_{tot}=$"+str(M_tot), fontsize=35)
        plt.show()
        
        sns.set(font_scale=1.1)
        vals = np.around(df.values,3)
        norm = plt.Normalize(vals.min(), vals.max()+0.1) # Variar en función de lo que queramos
        colours = plt.cm.gist_earth_r(norm(vals))
        
        # Tabla
        fig = plt.figure(figsize=(15,3))
        ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
        
        the_table = plt.table(cellText=np.around(rrll,3), rowLabels=df.index, colLabels=df.columns, 
                            colWidths = [0.08]*vals.shape[1], loc='center', 
                            cellColours=colours)
        plt.show()
        
    return R_tot, L_tot, T_central
