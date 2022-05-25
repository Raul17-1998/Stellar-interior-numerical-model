# =============================================================================
# ARCHIVO 1/3
# 
# Primero de los archivos a ejecutar.
# Este archivo contiene la función para encontrar el error relativo mínimo y
# la temperatura central óptima para los tres parámetros constantes y
# para los tres valores iniciales.
# Además, se incluye otra función que devuelve algo más de información,
# respecto de la primera, como gráficas, tablas, plots...
# =============================================================================

# Se importan las funciones necesarias

import numpy as np
from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
# import time

# start = time.time() # Para comenzar a calcular tiempo

print("----------------------------------------------------------------------")
print("--· INTRODUCIR DATOS DEL MODELO PARA CARGAR FUNCIONES Y PARÁMETROS ·--")
print("----------------------------------------------------------------------")

# Se cargan los datos para la generación de energía.

e1_pp = [10**-6.84,10**-6.04,10**-5.56,10**-5.02,10**-4.40]
e1_CN = [10**-22.2,10**-19.8,10**-17.1,10**-15.6,10**-12.5]
v_pp = [6.0,5.0,4.5,4.0,3.5]
v_CN = [20,18,16,15,13]
k = 1.380649e-16
N_A = 6.02214076e+23


# Se introducen los datos para los parámetros constantes
# y los valores iniciales de los que se parte

M_tot = float(input("La masa total es: "))
X = float(input("La proporción de H es: "))
Y = float(input("La proporción de He es: "))
R_tot = float(input("El radio total es: "))
L_tot = float(input("La luminosidad total es: "))
T_central = float(input("La temperatura central es: "))

Z = 1-X-Y 
mu = 1/(2*X+3/4*Y+1/2*Z)


# Se definen las funciones para la generación de energía 
# y el algoritmo Predictor-Corrector.

def generacion_energia_pp(T,P):
    rho = mu/(N_A*k)*P/T
    if T < 0.4:
        return 0
    elif T >= 0.4 and T < 0.6:
            return e1_pp[0]*X**2*rho*(10*T)**(v_pp[0])
    elif T >= 0.6 and T < 0.95:
            return e1_pp[1]*X**2*rho*(10*T)**(v_pp[1])
    elif T >= 0.95 and T < 1.2:
            return e1_pp[2]*X**2*rho*(10*T)**(v_pp[2])
    elif T >= 1.2 and T < 1.65:
            return e1_pp[3]*X**2*rho*(10*T)**(v_pp[3])
    else:
            return e1_pp[4]*X**2*rho*(10*T)**(v_pp[4])
        
def generacion_energia_CN(T,P):
    rho = mu/(N_A*k)*P/T
    if T < 1.2:
        return 0
    elif T >= 1.2 and T < 1.6:
        return e1_CN[0]*X*Z/3*rho*(10*T)**(v_CN[0])
    elif T >= 1.6 and T < 2.25:
        return e1_CN[1]*X*Z/3*rho*(10*T)**(v_CN[1])
    elif T >= 2.25 and T < 2.75:
        return e1_CN[2]*X*Z/3*rho*(10*T)**(v_CN[2])
    elif T >= 2.75 and T < 3.6:
        return e1_CN[3]*X*Z/3*rho*(10*T)**(v_CN[3])
    else:
        return e1_CN[4]*X*Z/3*rho*(10*T)**(v_CN[4])

def seleccionar_v(T,P):
    v = 0
    if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
        if T < 0.4:
            v = 0
        elif T >= 0.4 and T < 0.6:
            v = v_pp[0]
        elif T >= 0.6 and T < 0.95:
            v = v_pp[1]
        elif T >= 0.95 and T < 1.2:
            v = v_pp[2]
        elif T >= 1.2 and T < 1.65:
            v = v_pp[3]
        else:
            v = v_pp[4]
    if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
        if T < 1.2:
            v = 0
        elif T >= 1.2 and T < 1.6:
            v = v_CN[0]
        elif T >= 1.6 and T < 2.25:
            v = v_CN[1]
        elif T >= 2.25 and T < 2.75:
            v = v_CN[2]
        elif T >= 2.75 and T < 3.6:
            v = v_CN[3]
        else:
            v = v_CN[4]
    return v

def seleccionar_e1(T,P):
    e1 = 0
    if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
        if T < 0.4:
            e1 = 0
        elif T >= 0.4 and T < 0.6:
            e1 = e1_pp[0]
        elif T >= 0.6 and T < 0.95:
            e1 = e1_pp[1]
        elif T >= 0.95 and T < 1.2:
            e1 = e1_pp[2]
        elif T >= 1.2 and T < 1.65:
            e1 = e1_pp[3]
        else:
            e1 = e1_pp[4]
    if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
        if T < 1.2:
            e1 = 0
        elif T >= 1.2 and T < 1.6:
            e1 = e1_CN[0]
        elif T >= 1.6 and T < 2.25:
            e1 = e1_CN[1]
        elif T >= 2.25 and T < 2.75:
            e1 = e1_CN[2]
        elif T >= 2.75 and T < 3.6:
            e1 = e1_CN[3]
        else:
            e1 = e1_CN[4]
    return e1 

def pred_corr(x,y):
    if abs((x-y)/x) >= 0.0001:
        return False
    else:
        return True


# Primera de las funciones "grandes". A partir de los valores dados al inicio,
# devuelve el error relativo mínimo para ellos y la temperatura central óptima.

def funcion(M_tot,X,Y,R_tot,L_tot,T_central):
    Z = 1-X-Y 
    mu = 1/(2*X+3/4*Y+1/2*Z)

    
    # e1_pp,v_pp,e1_CN,v_CN = np.loadtxt("ciclos_energia.txt", delimiter = ' ',
                                       # skiprows=1, usecols=(0,1,2,3),
                                       # unpack=True)
    
    # =========================================================================
    # INTEGRACIÓN DESDE SUPERFICIE    
    # =========================================================================
    
    ### FASE A.1. ENVOLTURA RADIATIVA
    
    # Se crean los vectores para ir guardando los datos que se irán calculando
    # y se definen las ecuaciones necesarias.
    

    
    R_ini = 0.9*R_tot
    h = -R_ini/100
    
    k_i = np.linspace(0,100,101).astype('int') # 100 capas
    r = np.zeros(len(k_i))
    M = np.zeros(len(k_i))
    L = np.zeros(len(k_i))
    T = np.zeros(len(k_i))
    P = np.zeros(len(k_i))
    
    fase = ['-']*len(k_i)
    N = np.zeros(len(k_i))
    fase[0], fase[1], fase[2] = 'INICIO', 'INICIO', 'INICIO'
    
    def temperatura_rad(r):
        A_1 = 1.9022*mu*M_tot
        return A_1*(1/r-1/R_tot)
    
    def presion_rad(T):
        A_2 = 10.645*np.sqrt(1/(mu*Z*(1+X))*(M_tot/L_tot))
        return A_2*T**4.25
    
    r[0] = R_ini
    M[0] = M_tot
    L[0] = L_tot
    T[0] = temperatura_rad(r[0])
    P[0] = presion_rad(T[0])
    
    for i in range(2):
        r[i+1] = r[i]+h
        M[i+1] = M_tot
        L[i+1] = L_tot
        T[i+1] = temperatura_rad(r[i+1])
        P[i+1] = presion_rad(T[i+1])
        
    def dM_rad(P,T,r):
        C_m = 0.01523*mu
        return C_m*P/T*r**2
    
    def dP_rad(P,T,M,r):
        C_p = 8.084*mu
        return -C_p*P/T*M/r**2
    
    def dL_rad(P,T,r):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = 0.01845*e1*X1*X2*10**v*mu**2
        return C_l*P**2*T**(v-2)*r**2
    
    def dT_rad(P,T,L,r):
        C_t = 0.01679*Z*(1+X)*mu**2
        return -C_t*P**2/T**8.5*L/r**2
    
    der_M = np.zeros(len(k_i))
    der_P = np.zeros(len(k_i))
    der_L = np.zeros(len(k_i))
    der_T = np.zeros(len(k_i))
    
    for i in range(3):
        der_M[i] = 0 
        der_P[i] = dP_rad(P[i],T[i],M[i],r[i])
        der_L[i] = 0
        der_T[i] = dT_rad(P[i],T[i],L[i],r[i])
    
    def delta1_M(i):
        return h*der_M[i]-h*der_M[i-1]
    
    def delta2_M(i):
        return h*der_M[i]-2*h*der_M[i-1]+h*der_M[i-2]
    
    def delta1_P(i):
        return h*der_P[i]-h*der_P[i-1]
    
    def delta2_P(i):
        return h*der_P[i]-2*h*der_P[i-1]+h*der_P[i-2]
    
    def delta1_L(i):
        return h*der_L[i]-h*der_L[i-1]
    
    def delta2_L(i):
        return h*der_L[i]-2*h*der_L[i-1]+h*der_L[i-2]
    
    def delta1_T(i):
        return h*der_T[i]-h*der_T[i-1]
    
    def delta2_T(i):
        return h*der_T[i]-2*h*der_T[i-1]+h*der_T[i-2]
    
    def P_est(i):
        return P[i]+h*der_P[i]+1/2*delta1_P(i)+5/12*delta2_P(i)
    
    def T_est(i):
        return T[i]+h*der_T[i]+1/2*delta1_T(i)
    
    def M_cal(i):
        return M[i]+h*der_M[i+1]-1/2*delta1_M(i+1)
    
    def P_cal(i):
        return P[i]+h*der_P[i+1]-1/2*delta1_P(i+1)
    
    def L_cal(i):
        return L[i]+h*der_L[i+1]-1/2*delta1_L(i+1)-1/12*delta2_L(i+1)
    
    def T_cal(i):
        return T[i]+h*der_T[i+1]-1/2*delta1_T(i+1)
    
    
    # # Algoritmo A.1. Puesta en marcha.

    i = 2  # número de la última capa calculada
    
    C_m = 0.01523*mu
    C_p = 8.084*mu
    C_t = 0.01679*Z*(1+X)*mu**2
    
    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        P_estimada = P_est(i)
        T_estimada = T_est(i)
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
                M_calculada = M_cal(i)
                der_P[i+1] = -C_p*P_estimada/T_estimada*M_calculada/r[i+1]**2
                P_calculada = P_cal(i)
                if pred_corr(P_calculada,P_estimada) == True:
                    loop3 = False
                else:
                    P_estimada = P_calculada
            v = 0
            X1 = 0
            X2 = 0
            e1 = 0
            if generacion_energia_pp(T_estimada,P_estimada) > generacion_energia_CN(T_estimada,P_estimada):
                X1 = X
                X2 = X
            if generacion_energia_pp(T_estimada,P_estimada) < generacion_energia_CN(T_estimada,P_estimada):
                X1 = X
                X2 = 1/3*Z
            e1 = seleccionar_e1(T_estimada,P_estimada)
            v = seleccionar_v(T_estimada,P_estimada)
            der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_estimada**(v-2)*r[i+1]**2
            L_calculada = L_cal(i)
            der_T[i+1] = -C_t*P_calculada**2/T_estimada**8.5*L_calculada/r[i+1]**2
            T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada
        n = T_calculada/P_calculada*(h*der_P[i+1])/(h*der_T[i+1])
        if n <= 2.5:
            loop1 = False
            fase[i+1] = 'CONVEC'
            N[i+1] = n
        else:
            # print("Pasamos a la capa", i+2)
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'RADIAT'
            N[i+1] = n
            i += 1
    
    
    # Se guardan los datos para trabajar más cómodamente más tarde.
    
    r_rad = r[:fase.index('CONVEC')]
    capas_rad = k_i[:fase.index('CONVEC')]
    
    
    ### FASE A.2. NÚCLEO CONVECTIVO
    
    K = P_cal(i)/T_cal(i)**2.5
    
    def dM_convec(T,r):
        C_m = 0.01523*mu
        return C_m*K*T**1.5*r**2
    
    def dP_convec(T,M,r):
        C_p = 8.084*mu
        return -C_p*K*T**1.5*M/r**2
    
    def dL_convec(P,T,r):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = 0.01845*e1*X1*X2*10**v*mu**2
        return C_l*K**2*T**(3+v)*r**2
    
    def dT_convec(M,r):
        C_t = 3.234*mu
        return -C_t*M/r**2
    
    der_M[i+1] = dM_convec(T[i],r[i]) 
    der_P[i+1] = dP_convec(T[i],M[i],r[i])
    der_L[i+1] = dL_convec(P[i],T[i],r[i])
    der_T[i+1] = dT_convec(M[i],r[i])
    
    
    # # Algoritmo A.2.
    
    C_m = 0.01523*mu
    C_t = 3.234*mu
    
    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        T[i+1] = T_est(i)
        T_estimada = T[i+1]
        loop2 = True
        while loop2:
            P[i+1] = K*T_estimada**2.5
            P_estimada = P[i+1]
            der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
            M_calculada = M_cal(i)
            if r[i+1] == 0:
                T_calculada = T_estimada
            elif r[i+1] > 0:
                der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada  
        P_calculada = K*T_calculada**2.5
        v = 0
        X1 = 0
        X2 = 0
        e1 = 0
        if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
            X1 = X
            X2 = X
        if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T_calculada,P_calculada)
        v = seleccionar_v(T_calculada,P_calculada)
        der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
        L_calculada = L_cal(i)
        if r[i+1] > 10**-6:
            loop1 = True
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
            # print("Pasamos a la capa", i+2)
            i += 1
        else:
            loop1 = False
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
    
    
    # # Valores en la frontera (desde arriba)
    
    from scipy.interpolate import lagrange
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [r[len(r_rad)-1],r[len(r_rad)]])
    r_front = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [P[len(r_rad)-1],P[len(r_rad)]])
    P_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [T[len(r_rad)-1],T[len(r_rad)]])
    T_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [L[len(r_rad)-1],L[len(r_rad)]])
    L_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [M[len(r_rad)-1],M[len(r_rad)]])
    M_front_down = pol(2.50)
    
    
    # =========================================================================
    # INTEGRACIÓN DESDE CENTRO  
    # =========================================================================

    # Se crean los vectores para ir guardando los datos que se irán calculando
    # y se definen las ecuaciones necesarias.

    k_i = np.arange(0,102-len(capas_rad),1) # capas
    r = np.zeros(len(k_i))
    M = np.zeros(len(k_i))
    L = np.zeros(len(k_i))
    T = np.zeros(len(k_i))
    P = np.zeros(len(k_i))
    
    fase = ['-']*len(k_i)
    fase[0], fase[1], fase[2] = 'CENTRO', 'CENTRO', 'CENTRO'
    
    def masa_convec(r):
        return 0.005077*mu*K*T_central**1.5*r**3
    
    def luminosidad_convec(r,T,P):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = e1*X1*X2*10**v*mu**2
        return 0.006150*C_l*K**2*T_central**(3+v)*r**3
    
    def temperatura_convec(r):
        return T_central-0.008207*mu**2*K*T_central**1.5*r**2
    
    def presion_convec(r):
        return K*temperatura_convec(r)**2.5
    
    h = R_ini/100
    r[0] = 0
    T[0] = T_central
    
    for i in range(2):
        r[i+1] = r[i]+h
        T[i+1] = temperatura_convec(r[i+1])
    for i in range(3):
        M[i] = masa_convec(r[i])
        P[i] = presion_convec(r[i])
        L[i] = luminosidad_convec(r[i],T[i],P[i])
    
    der_M = np.zeros(len(k_i))
    der_P = np.zeros(len(k_i))
    der_L = np.zeros(len(k_i))
    der_T = np.zeros(len(k_i))
    
    for i in range(3):
        der_M[i] = dM_convec(T[i],r[i]) 
        der_L[i] = dL_convec(P[i],T[i],r[i])
    for i in range(2):
        der_T[i+1] = dT_convec(M[i+1],r[i+1])
    
    der_M[i] = dM_convec(T[i],r[i]) 
    der_L[i] = dL_convec(P[i],T[i],r[i])
    der_T[i] = dT_convec(M[i],r[i])
    
    
    # # Algoritmo A.2.
    i = 2
    
    C_m = 0.01523*mu
    C_t = 3.234*mu
        
    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        T[i+1] = T_est(i)
        T_estimada = T[i+1]
        loop2 = True
        while loop2:
            P[i+1] = K*T_estimada**2.5
            P_estimada = P[i+1]
            der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
            M_calculada = M_cal(i)
            if r[i+1] <= 10**-20:
                T_calculada = T_estimada
            elif r[i+1] > 0:
                der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada  
        P_calculada = K*T_calculada**2.5
        v = 0
        X1 = 0
        X2 = 0
        e1 = 0
        if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
            X1 = X
            X2 = X
        if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T_calculada,P_calculada)
        v = seleccionar_v(T_calculada,P_calculada)
        der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
        L_calculada = L_cal(i)
        if abs(r[i+1]-r_rad[len(r_rad)-1]) > 10**-6:
            loop1 = True
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
            # print("Pasamos a la capa", i+2)
            i += 1
        elif abs(r[i+1]-r_rad[len(r_rad)-1]) <= 10**-6:
            loop1 = False
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
    
    
    # # Valores en la frontera (desde abajo)
    
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [P[len(fase)-2],P[len(fase)-1]])
    P_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [T[len(fase)-2],T[len(fase)-1]])
    T_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [L[len(fase)-2],L[len(fase)-1]])
    L_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [M[len(fase)-2],M[len(fase)-1]])
    M_front_up = pol(r_front)
    
    
    # =========================================================================
    # AJUSTE ERROR MÍNIMO  
    # =========================================================================
    
    # Se calcula el error relativo total. Variando la temperatura central,
    # se buscará el error relativo total mínimo. Se utilizarán bucles para
    # generar intervalos cada vez más pequeños alrededor de la temperatura
    # central óptima obtenida en el bucle anterior.
    
    # Basta con calcular la zona convectiva, pues son sus ecuaciones las
    # únicas que dependen de la temperatura central.
    
    error_rel_P = abs((P_front_down-P_front_up)/P_front_down)
    error_rel_T = abs((T_front_down-T_front_up)/T_front_down)
    error_rel_L = abs((L_front_down-L_front_up)/L_front_down)
    error_rel_M = abs((M_front_down-M_front_up)/M_front_down)
    error_rel_tot = np.sqrt(error_rel_P**2+error_rel_T**2+error_rel_L**2+error_rel_M**2)
    
    k_i = np.arange(0,102-len(capas_rad),1) #capas
    h = R_ini/100
       
    for s in range(4):
        r = np.zeros(len(k_i))
        M = np.zeros(len(k_i))
        L = np.zeros(len(k_i))
        T = np.zeros(len(k_i))
        P = np.zeros(len(k_i))
        
        TT = np.linspace(T_central-0.1**s,T_central+0.1**s,500)
        error_relativo = np.zeros(len(TT))
            
        for j in range(len(TT)):
            T_central = TT[j]
            r[0] = 0
            T[0] = T_central
    
            for i in range(2):
                r[i+1] = r[i]+h
                T[i+1] = temperatura_convec(r[i+1])
            for i in range(3):
                M[i] = masa_convec(r[i])
                P[i] = presion_convec(r[i])
                L[i] = luminosidad_convec(r[i],T[i],P[i])
            der_M = np.zeros(len(k_i))
            der_P = np.zeros(len(k_i))
            der_L = np.zeros(len(k_i))
            der_T = np.zeros(len(k_i))
            der_T[0] =  0
            for i in range(3):
                der_M[i] = dM_convec(T[i],r[i]) 
                der_L[i] = dL_convec(P[i],T[i],r[i])
            for i in range(2):
                der_T[i+1] = dT_convec(M[i+1],r[i+1])
                
            i = 2
    
            C_m = 0.01523*mu
            C_t = 3.234*mu
    
            loop1 = True
            while loop1:
                # print("\n*Calculando capa número:", i+1, "\n")
                r[i+1] = r[i]+h
                T[i+1] = T_est(i)
                T_estimada = T[i+1]
                loop2 = True
                while loop2:
                    P[i+1] = K*T_estimada**2.5
                    P_estimada = P[i+1]
                    der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
                    M_calculada = M_cal(i)
                    if r[i+1] == 0:
                        T_calculada = T_estimada
                    elif r[i+1] > 0:
                        der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                        T_calculada = T_cal(i)
                    if pred_corr(T_calculada,T_estimada) == True:
                        loop2 = False
                    else:
                        T_estimada = T_calculada  
                P_calculada = K*T_calculada**2.5
                v = 0
                X1 = 0
                X2 = 0
                e1 = 0
                if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
                    X1 = X
                    X2 = X
                if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
                    X1 = X
                    X2 = 1/3*Z
                e1 = seleccionar_e1(T_calculada,P_calculada)
                v = seleccionar_v(T_calculada,P_calculada)
                der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
                L_calculada = L_cal(i)
                if abs(r[i+1]-r_rad[len(r_rad)-1]) > 10**-6:
                    loop1 = True
                    P[i+1] = P_calculada
                    M[i+1] = M_calculada
                    T[i+1] = T_calculada
                    L[i+1] = L_calculada
                    # print("Pasamos a la capa", i+2)
                    i += 1
                elif abs(r[i+1]-r_rad[len(r_rad)-1]) <= 10**-6:
                    loop1 = False
                    P[i+1] = P_calculada
                    M[i+1] = M_calculada
                    T[i+1] = T_calculada
                    L[i+1] = L_calculada
    
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [P[len(r)-2],P[len(r)-1]])
            P_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [T[len(r)-2],T[len(r)-1]])
            T_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [L[len(r)-2],L[len(r)-1]])
            L_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [M[len(r)-2],M[len(r)-1]])
            M_front_up = pol(r_front)
    
            error_rel_P = abs((P_front_down-P_front_up)/P_front_down)
            error_rel_T = abs((T_front_down-T_front_up)/T_front_down)
            error_rel_L = abs((L_front_down-L_front_up)/L_front_down)
            error_rel_M = abs((M_front_down-M_front_up)/M_front_down)
            error_rel_tot = np.sqrt(error_rel_P**2+error_rel_T**2+error_rel_L**2+error_rel_M**2)
    
            error_relativo[j] = error_rel_tot*100
            
        mi = np.where(error_relativo == min(error_relativo))[0][0]
        minimo_error = error_relativo[mi]
        T_central = TT[mi]
 
    return minimo_error, T_central


# Segunda función "grande". Casi idéntica que la anterior, pero incluye
# información adicional y omite ciertos cálculos.

def funcion_graficas(M_tot,X,Y,R_tot,L_tot,T_central):
    Z = 1-X-Y 
    mu = 1/(2*X+3/4*Y+1/2*Z)
    
    # =========================================================================
    # INTEGRACIÓN DESDE SUPERFICIE  
    # =========================================================================

    R_ini = 0.9*R_tot
    h = -R_ini/100
    
    k_i = np.linspace(0,100,101).astype('int') # 100 capas
    r = np.zeros(len(k_i))
    M = np.zeros(len(k_i))
    L = np.zeros(len(k_i))
    T = np.zeros(len(k_i))
    P = np.zeros(len(k_i))
    op = np.zeros(len(k_i))
    
    ciclo_energia = ['-']*len(k_i)
    fase = ['-']*len(k_i)
    N = np.zeros(len(k_i))
    fase[0], fase[1], fase[2] = 'INICIO', 'INICIO', 'INICIO'

    def temperatura_rad(r):
        A_1 = 1.9022*mu*M_tot
        return A_1*(1/r-1/R_tot)
    
    def presion_rad(T):
        A_2 = 10.645*np.sqrt(1/(mu*Z*(1+X))*(M_tot/L_tot))
        return A_2*T**4.25

    # def opacidad(P,T):
        # rho = mu/(N_A*k)*P/(T)
        # tg = 9.9e29
        # return 4.34e25/tg*Z*(1+X)*rho/(T**3.5)

    def opacidad(P,T):
        rho = mu*P/(T*N_A*k)
        return 4.34e25*Z*(1+X)*rho/(T**3.5)*5040/T

    r[0] = R_ini
    M[0] = M_tot
    L[0] = L_tot
    T[0] = temperatura_rad(r[0])
    P[0] = presion_rad(T[0])

    for i in range(2):
        r[i+1] = r[i]+h
        M[i+1] = M_tot
        L[i+1] = L_tot
        T[i+1] = temperatura_rad(r[i+1])
        P[i+1] = presion_rad(T[i+1])
        
    for i in range(3):
        op[i] = opacidad(P[i],T[i])

    def dM_rad(P,T,r):
        C_m = 0.01523*mu
        return C_m*P/T*r**2
    
    def dP_rad(P,T,M,r):
        C_p = 8.084*mu
        return -C_p*P/T*M/r**2
    
    def dL_rad(P,T,r):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = 0.01845*e1*X1*X2*10**v*mu**2
        return C_l*P**2*T**(v-2)*r**2
    
    def dT_rad(P,T,L,r):
        C_t = 0.01679*Z*(1+X)*mu**2
        return -C_t*P**2/T**8.5*L/r**2


    der_M = np.zeros(len(k_i))
    der_P = np.zeros(len(k_i))
    der_L = np.zeros(len(k_i))
    der_T = np.zeros(len(k_i))

    for i in range(3):
        der_M[i] = 0 
        der_P[i] = dP_rad(P[i],T[i],M[i],r[i])
        der_L[i] = 0
        der_T[i] = dT_rad(P[i],T[i],L[i],r[i])

    def delta1_M(i):
        return h*der_M[i]-h*der_M[i-1]

    def delta2_M(i):
        return h*der_M[i]-2*h*der_M[i-1]+h*der_M[i-2]

    def delta1_P(i):
        return h*der_P[i]-h*der_P[i-1]

    def delta2_P(i):
        return h*der_P[i]-2*h*der_P[i-1]+h*der_P[i-2]

    def delta1_L(i):
        return h*der_L[i]-h*der_L[i-1]

    def delta2_L(i):
        return h*der_L[i]-2*h*der_L[i-1]+h*der_L[i-2]

    def delta1_T(i):
        return h*der_T[i]-h*der_T[i-1]

    def delta2_T(i):
        return h*der_T[i]-2*h*der_T[i-1]+h*der_T[i-2]

    def P_est(i):
        return P[i]+h*der_P[i]+1/2*delta1_P(i)+5/12*delta2_P(i)

    def T_est(i):
        return T[i]+h*der_T[i]+1/2*delta1_T(i)

    def M_cal(i):
        return M[i]+h*der_M[i+1]-1/2*delta1_M(i+1)

    def P_cal(i):
        return P[i]+h*der_P[i+1]-1/2*delta1_P(i+1)

    def L_cal(i):
        return L[i]+h*der_L[i+1]-1/2*delta1_L(i+1)-1/12*delta2_L(i+1)

    def T_cal(i):
        return T[i]+h*der_T[i+1]-1/2*delta1_T(i+1)

    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    i = 2  # número de la última capa calculada

    C_m = 0.01523*mu
    C_p = 8.084*mu
    C_t = 0.01679*Z*(1+X)*mu**2
    ciclo = 0

    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        P_estimada = P_est(i)
        T_estimada = T_est(i)
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
                M_calculada = M_cal(i)
                der_P[i+1] = -C_p*P_estimada/T_estimada*M_calculada/r[i+1]**2
                P_calculada = P_cal(i)
                if pred_corr(P_calculada,P_estimada) == True:
                    loop3 = False
                else:
                    P_estimada = P_calculada
            v = 0
            X1 = 0
            X2 = 0
            e1 = 0
            if generacion_energia_pp(T_estimada,P_estimada) > generacion_energia_CN(T_estimada,P_estimada):
                ciclo = 1
                X1 = X
                X2 = X
            if generacion_energia_pp(T_estimada,P_estimada) < generacion_energia_CN(T_estimada,P_estimada):
                ciclo = 2
                X1 = X
                X2 = 1/3*Z
            e1 = seleccionar_e1(T_estimada,P_estimada)
            v = seleccionar_v(T_estimada,P_estimada)
            der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_estimada**(v-2)*r[i+1]**2
            L_calculada = L_cal(i)
            der_T[i+1] = -C_t*P_calculada**2/T_estimada**8.5*L_calculada/r[i+1]**2
            T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada
        n = T_calculada/P_calculada*(h*der_P[i+1])/(h*der_T[i+1])
        if n <= 2.5:
            loop1 = False
            fase[i+1] = 'CONVEC'
            N[i+1] = n
        else:
            # pasamos a la siguiente capa
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'RADIAT'
            N[i+1] = n
            op[i+1] = opacidad(P[i+1],T[i+1])
            i += 1

    r_rad = r[:fase.index('CONVEC')]
    P_rad = P[:fase.index('CONVEC')]
    T_rad = T[:fase.index('CONVEC')]
    L_rad = L[:fase.index('CONVEC')]
    M_rad = M[:fase.index('CONVEC')]
    N_rad = N[:fase.index('CONVEC')]
    capas_rad = k_i[:fase.index('CONVEC')]
    fase_rad = fase[:fase.index('CONVEC')]
    ciclo_rad = ciclo_energia[:fase.index('CONVEC')]
    op_rad = op[:fase.index('CONVEC')]


    K = P_cal(i)/T_cal(i)**2.5

    def dM_convec(T,r):
        C_m = 0.01523*mu
        return C_m*K*T**1.5*r**2

    def dP_convec(T,M,r):
        C_p = 8.084*mu
        return -C_p*K*T**1.5*M/r**2

    def dL_convec(P,T,r):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = 0.01845*e1*X1*X2*10**v*mu**2
        return C_l*K**2*T**(3+v)*r**2

    def dT_convec(M,r):
        C_t = 3.234*mu
        return -C_t*M/r**2

    der_M[i+1] = dM_convec(T[i],r[i]) 
    der_P[i+1] = dP_convec(T[i],M[i],r[i])
    der_L[i+1] = dL_convec(P[i],T[i],r[i])
    der_T[i+1] = dT_convec(M[i],r[i])

    C_m = 0.01523*mu
    C_t = 3.234*mu

    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        T[i+1] = T_est(i)
        T_estimada = T[i+1]
        loop2 = True
        while loop2:
            P[i+1] = K*T_estimada**2.5
            P_estimada = P[i+1]
            der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
            M_calculada = M_cal(i)
            if r[i+1] == 0:
                T_calculada = T_estimada
            elif r[i+1] > 0:
                der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada  
        P_calculada = K*T_calculada**2.5
        v = 0
        X1 = 0
        X2 = 0
        e1 = 0
        if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 1
            X1 = X
            X2 = X
        if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 2
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T_calculada,P_calculada)
        v = seleccionar_v(T_calculada,P_calculada)
        der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
        L_calculada = L_cal(i)
        if r[i+1] > 10**-6:
            loop1 = True
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
            # print("Pasamos a la capa", i+2)
            i += 1
        else:
            loop1 = False
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'

    from scipy.interpolate import lagrange
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [r[len(r_rad)-1],r[len(r_rad)]])
    r_front = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [P[len(r_rad)-1],P[len(r_rad)]])
    P_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [T[len(r_rad)-1],T[len(r_rad)]])
    T_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [L[len(r_rad)-1],L[len(r_rad)]])
    L_front_down = pol(2.50)
    pol = lagrange([N[len(r_rad)-1],N[len(r_rad)]],
                   [M[len(r_rad)-1],M[len(r_rad)]])
    M_front_down = pol(2.50)


    # =========================================================================
    # INTEGRACIÓN DESDE CENTRO  
    # =========================================================================

    k_i = np.arange(0,102-len(capas_rad),1) # capas
    r = np.zeros(len(k_i))
    M = np.zeros(len(k_i))
    L = np.zeros(len(k_i))
    T = np.zeros(len(k_i))
    P = np.zeros(len(k_i))


    ciclo_energia = ['-']*len(k_i)
    fase = ['-']*len(k_i)
    fase[0], fase[1], fase[2] = 'CENTRO', 'CENTRO', 'CENTRO'

    def masa_convec(r):
        return 0.005077*mu*K*T_central**1.5*r**3

    def luminosidad_convec(r,T,P):
        X1 = 0
        X2 = 0
        if generacion_energia_pp(T,P) > generacion_energia_CN(T,P):
            X1 = X
            X2 = X
        if generacion_energia_pp(T,P) < generacion_energia_CN(T,P):
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T,P)
        v = seleccionar_v(T,P)
        C_l = e1*X1*X2*10**v*mu**2
        return 0.006150*C_l*K**2*T_central**(3+v)*r**3

    def temperatura_convec(r):
        return T_central-0.008207*mu**2*K*T_central**1.5*r**2

    def presion_convec(r):
        return K*temperatura_convec(r)**2.5

    h = R_ini/100
    r[0] = 0
    T[0] = T_central

    for i in range(2):
        r[i+1] = r[i]+h
        T[i+1] = temperatura_convec(r[i+1])
    for i in range(3):
        M[i] = masa_convec(r[i])
        P[i] = presion_convec(r[i])
        L[i] = luminosidad_convec(r[i],T[i],P[i])

    der_M = np.zeros(len(k_i))
    der_P = np.zeros(len(k_i))
    der_L = np.zeros(len(k_i))
    der_T = np.zeros(len(k_i))

    for i in range(3):
        der_M[i] = dM_convec(T[i],r[i]) 
        # der_P[i] = dP_convec(T[i],M[i],r[i])
        der_L[i] = dL_convec(P[i],T[i],r[i])
    for i in range(2):
        der_T[i+1] = dT_convec(M[i+1],r[i+1])

    i = 2

    C_m = 0.01523*mu
    C_t = 3.234*mu

    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        T[i+1] = T_est(i)
        T_estimada = T[i+1]
        loop2 = True
        while loop2:
            P[i+1] = K*T_estimada**2.5
            P_estimada = P[i+1]
            der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
            M_calculada = M_cal(i)
            if r[i+1] == 0:
                T_calculada = T_estimada
            elif r[i+1] > 0:
                der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada  
        P_calculada = K*T_calculada**2.5
        v = 0
        X1 = 0
        X2 = 0
        e1 = 0
        if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 1
            X1 = X
            X2 = X
        if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 2
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T_calculada,P_calculada)
        v = seleccionar_v(T_calculada,P_calculada)
        der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
        L_calculada = L_cal(i)
        if abs(r[i+1]-r_rad[len(r_rad)-1]) > 10**-6:
            loop1 = True
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'
            # print("Pasamos a la capa", i+2)
            i += 1
        elif abs(r[i+1]-r_rad[len(r_rad)-1]) <= 10**-6:
            loop1 = False
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            fase[i+1] = 'CONVEC'

    r_convec = r[:len(fase)-1]
    P_convec = P[:len(fase)-1]
    T_convec = T[:len(fase)-1]
    L_convec = L[:len(fase)-1]
    M_convec = M[:len(fase)-1]
    capas_convec = k_i[:len(fase)-1]
    fase_convec = fase[:len(fase)-1]
    ciclo_convec = ciclo_energia[:len(fase)-1]

    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [P[len(fase)-2],P[len(fase)-1]])
    P_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [T[len(fase)-2],T[len(fase)-1]])
    T_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [L[len(fase)-2],L[len(fase)-1]])
    L_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [M[len(fase)-2],M[len(fase)-1]])
    M_front_up = pol(r_front)


    # =========================================================================
    # AJUSTE ERROR MÍNIMO  
    # =========================================================================

    error_rel_P = abs((P_front_down-P_front_up)/P_front_down)
    error_rel_T = abs((T_front_down-T_front_up)/T_front_down)
    error_rel_L = abs((L_front_down-L_front_up)/L_front_down)
    error_rel_M = abs((M_front_down-M_front_up)/M_front_down)
    error_rel_tot = np.sqrt(error_rel_P**2+error_rel_T**2+error_rel_L**2+error_rel_M**2)

    k_i = np.arange(0,102-len(capas_rad),1) #capas
    h = R_ini/100

    for s in range(4):
        r = np.zeros(len(k_i))
        M = np.zeros(len(k_i))
        L = np.zeros(len(k_i))
        T = np.zeros(len(k_i))
        P = np.zeros(len(k_i))

        TT = np.linspace(T_central-0.1**s,T_central+0.1**s,500)
        error_relativo = np.zeros(len(TT))

        for j in range(len(TT)):
            T_central = TT[j]
            r[0] = 0
            T[0] = T_central

            for i in range(2):
                r[i+1] = r[i]+h
                T[i+1] = temperatura_convec(r[i+1])
            for i in range(3):
                M[i] = masa_convec(r[i])
                P[i] = presion_convec(r[i])
                L[i] = luminosidad_convec(r[i],T[i],P[i])
            der_M = np.zeros(len(k_i))
            der_P = np.zeros(len(k_i))
            der_L = np.zeros(len(k_i))
            der_T = np.zeros(len(k_i))
            der_T[0] =  0
            for i in range(3):
                der_M[i] = dM_convec(T[i],r[i]) 
                der_L[i] = dL_convec(P[i],T[i],r[i])
            for i in range(2):
                der_T[i+1] = dT_convec(M[i+1],r[i+1])

            i = 2

            C_m = 0.01523*mu
            C_t = 3.234*mu

            loop1 = True
            while loop1:
                # print("\n*Calculando capa número:", i+1, "\n")
                r[i+1] = r[i]+h
                T[i+1] = T_est(i)
                T_estimada = T[i+1]
                loop2 = True
                while loop2:
                    P[i+1] = K*T_estimada**2.5
                    P_estimada = P[i+1]
                    der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
                    M_calculada = M_cal(i)
                    if r[i+1] == 0:
                        T_calculada = T_estimada
                    elif r[i+1] > 0:
                        der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                        T_calculada = T_cal(i)
                    if pred_corr(T_calculada,T_estimada) == True:
                        loop2 = False
                    else:
                        T_estimada = T_calculada  
                P_calculada = K*T_calculada**2.5
                v = 0
                X1 = 0
                X2 = 0
                e1 = 0
                if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
                    X1 = X
                    X2 = X
                if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
                    X1 = X
                    X2 = 1/3*Z
                e1 = seleccionar_e1(T_calculada,P_calculada)
                v = seleccionar_v(T_calculada,P_calculada)
                der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
                L_calculada = L_cal(i)
                if abs(r[i+1]-r_rad[len(r_rad)-1]) > 10**-6:
                    loop1 = True
                    P[i+1] = P_calculada
                    M[i+1] = M_calculada
                    T[i+1] = T_calculada
                    L[i+1] = L_calculada
                    # print("Pasamos a la capa", i+2)
                    i += 1
                elif abs(r[i+1]-r_rad[len(r_rad)-1]) <= 10**-6:
                    loop1 = False
                    P[i+1] = P_calculada
                    M[i+1] = M_calculada
                    T[i+1] = T_calculada
                    L[i+1] = L_calculada

            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [P[len(r)-2],P[len(r)-1]])
            P_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [T[len(r)-2],T[len(r)-1]])
            T_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [L[len(r)-2],L[len(r)-1]])
            L_front_up = pol(r_front)
            pol = lagrange([r[len(r)-2],r[len(r)-1]],
                           [M[len(r)-2],M[len(r)-1]])
            M_front_up = pol(r_front)

            error_rel_P = abs((P_front_down-P_front_up)/P_front_down)
            error_rel_T = abs((T_front_down-T_front_up)/T_front_down)
            error_rel_L = abs((L_front_down-L_front_up)/L_front_down)
            error_rel_M = abs((M_front_down-M_front_up)/M_front_down)
            error_rel_tot = np.sqrt(error_rel_P**2+error_rel_T**2+error_rel_L**2+error_rel_M**2)

            error_relativo[j] = error_rel_tot*100

        minimo_error = min(error_relativo)

        for i in range(len(TT)):
            if error_relativo[i] == minimo_error:
                T_central = TT[i]

    radio = np.append(r_rad,r_convec[::-1])
    presion = np.append(P_rad,P_convec[::-1])
    temperatura = np.append(T_rad,T_convec[::-1])
    luminosidad = np.append(L_rad,L_convec[::-1])
    masa = np.append(M_rad,M_convec[::-1])



    # =========================================================================
    # MODELO COMPLETO
    # =========================================================================

    k_i = np.arange(0,102-len(capas_rad),1) # capas
    r = np.zeros(len(k_i))
    M = np.zeros(len(k_i))
    L = np.zeros(len(k_i))
    T = np.zeros(len(k_i))
    P = np.zeros(len(k_i))
    op = np.zeros(len(k_i))

    ciclo_energia = ['-']*len(k_i)
    fase = ['-']*len(k_i)
    fase[0], fase[1], fase[2] = 'CENTRO', 'CENTRO', 'CENTRO'

    h = R_ini/100
    r[0] = 0
    T[0] = T_central

    for i in range(2):
        r[i+1] = r[i]+h
        T[i+1] = temperatura_convec(r[i+1])
    for i in range(3):
        M[i] = masa_convec(r[i])
        P[i] = presion_convec(r[i])
        L[i] = luminosidad_convec(r[i],T[i],P[i])
        op[i] = opacidad(P[i],T[i])

    der_M = np.zeros(len(k_i))
    der_P = np.zeros(len(k_i))
    der_L = np.zeros(len(k_i))
    der_T = np.zeros(len(k_i))

    for i in range(3):
        der_M[i] = dM_convec(T[i],r[i]) 
        # der_P[i] = dP_convec(T[i],M[i],r[i])
        der_L[i] = dL_convec(P[i],T[i],r[i])
    for i in range(2):
        der_T[i+1] = dT_convec(M[i+1],r[i+1])

    i = 2

    C_m = 0.01523*mu
    C_t = 3.234*mu

    loop1 = True
    while loop1:
        # print("\n*Calculando capa número:", i+1, "\n")
        r[i+1] = r[i]+h
        T[i+1] = T_est(i)
        T_estimada = T[i+1]
        loop2 = True
        while loop2:
            P[i+1] = K*T_estimada**2.5
            P_estimada = P[i+1]
            der_M[i+1] = C_m*P_estimada/T_estimada*r[i+1]**2
            M_calculada = M_cal(i)
            if r[i+1] == 0:
                T_calculada = T_estimada
            elif r[i+1] > 0:
                der_T[i+1] = -C_t*M_calculada/r[i+1]**2
                T_calculada = T_cal(i)
            if pred_corr(T_calculada,T_estimada) == True:
                loop2 = False
            else:
                T_estimada = T_calculada  
        P_calculada = K*T_calculada**2.5
        v = 0
        X1 = 0
        X2 = 0
        e1 = 0
        if generacion_energia_pp(T_calculada,P_calculada) > generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 1
            X1 = X
            X2 = X
        if generacion_energia_pp(T_calculada,P_calculada) < generacion_energia_CN(T_calculada,P_calculada):
            ciclo = 2
            X1 = X
            X2 = 1/3*Z
        e1 = seleccionar_e1(T_calculada,P_calculada)
        v = seleccionar_v(T_calculada,P_calculada)
        der_L[i+1] = 0.01845*e1*X1*X2*10**v*mu**2*P_calculada**2*T_calculada**(v-2)*r[i+1]**2
        L_calculada = L_cal(i)
        if abs(r[i+1]-r_rad[len(r_rad)-1]) > 10**-6:
            loop1 = True
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            op[i+1] = opacidad(P[i+1],T[i+1])
            fase[i+1] = 'CONVEC'
            # print("Pasamos a la capa", i+2)
            i += 1
        elif abs(r[i+1]-r_rad[len(r_rad)-1]) <= 10**-6:
            loop1 = False
            if ciclo == 1:
                ciclo_energia[i+1] = 'PP'
            elif ciclo == 2:
                ciclo_energia[i+1] = 'CN'
            P[i+1] = P_calculada
            M[i+1] = M_calculada
            T[i+1] = T_calculada
            L[i+1] = L_calculada
            op[i+1] = opacidad(P[i+1],T[i+1])
            fase[i+1] = 'CONVEC'    

    r_convec = r[:len(fase)-1]
    P_convec = P[:len(fase)-1]
    T_convec = T[:len(fase)-1]
    L_convec = L[:len(fase)-1]
    M_convec = M[:len(fase)-1]
    capas_convec = k_i[:len(fase)-1]
    fase_convec = fase[:len(fase)-1]
    ciclo_convec = ciclo_energia[:len(fase)-1]
    op_convec = op[:len(fase)-1]

    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [P[len(fase)-2],P[len(fase)-1]])
    P_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [T[len(fase)-2],T[len(fase)-1]])
    T_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [L[len(fase)-2],L[len(fase)-1]])
    L_front_up = pol(r_front)
    pol = lagrange([r[len(fase)-2],r[len(fase)-1]],
                   [M[len(fase)-2],M[len(fase)-1]])
    M_front_up = pol(r_front)    

    error_rel_P = abs((P_front_down-P_front_up)/P_front_down)
    error_rel_T = abs((T_front_down-T_front_up)/T_front_down)
    error_rel_L = abs((L_front_down-L_front_up)/L_front_down)
    error_rel_M = abs((M_front_down-M_front_up)/M_front_down)
    error_rel_tot = np.sqrt(error_rel_P**2+error_rel_T**2+error_rel_L**2+error_rel_M**2)    

    ciclo_energia = np.append(ciclo_rad,ciclo_convec[::-1])
    fase = np.append(fase_rad,fase_convec[::-1])
    radio = np.append(r_rad,r_convec[::-1])
    presion = np.append(P_rad,P_convec[::-1])
    temperatura = np.append(T_rad,T_convec[::-1])
    luminosidad = np.append(L_rad,L_convec[::-1])
    masa = np.append(M_rad,M_convec[::-1])
    opa = np.append(op_rad,op_convec[::-1])
    k_i = np.arange(0,len(radio),1)
    N = np.zeros_like(radio)
    for i in range(len(N_rad)):
        N[i] = N_rad[i]    

    ### TABLAS
    # datos = np.transpose([ciclo_energia, fase, k_i, radio, presion, temperatura, luminosidad, masa, N])
    titulos = ['Energía', 'Fase', 'Capa', 'Radio', 'Presión', 'Temperatura', 'Luminosidad', 'Masa', 'n+1']
    # datos = np.transpose([ciclo_energia, fase, k_i, radio, presion, temperatura, luminosidad, masa, opa, N])
    # titulos = ['Energía', 'Fase', 'Capa', 'Radio', 'Presión', 'Temperatura', 'Luminosidad', 'Masa', 'Opacidad', 'n+1']
    ## print(tabulate(datos, headers=titulos, tablefmt='plain', stralign='center', floatfmt='.7f'))
    # print(tabulate(datos, headers=titulos, tablefmt='fancy_grid', stralign='center', floatfmt='.7f'))    
    
    data = {'Energía': ciclo_energia,
            'Fase': fase,
            'Capa': k_i,
            'Radio': radio,
            'Presión': presion,
            'Temperatura': temperatura,
            'Luminosidad': luminosidad,
            'Masa': masa,
            # 'Opacidad': opa,
            'n+1': N}

    print(tabulate(data, headers=titulos, tablefmt='plain', stralign='center', floatfmt='.7f'))

    d = {'Energía': ciclo_energia,
         'Fase': fase,
         'Capa': k_i,
         'Radio': radio,
         'Presión': presion,
         'Temperatura': temperatura,
         'Luminosidad': luminosidad,
         'Masa': masa,
         # 'Opacidad': opa,
         'n+1': N}
    
    df = pd.DataFrame(d)
    df.to_csv('param_finales.csv', sep='\t', index=False)
    
    rr = radio/max(radio)
      
    print("El peso molecular medio es ", mu)
    print("La frontera entre fases se encuentra para r=", r_front)
    print("La temperatura central es ", T_central)
    print("El error relativo total (%) es ", error_rel_tot*100)
    
    # Plot sin normalizar
    
    fig = plt.figure(figsize=(15,9))
    fig.patch.set_facecolor('xkcd:white')
    ax = plt.gca()
    ax.set_facecolor('xkcd:white')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')    
    size = np.ones_like(radio)*30
    plt.scatter(radio,presion, s = size, color='black', label='Presión')
    plt.scatter(radio,temperatura, s = size, color='orange', label='Temperatura')
    plt.scatter(radio,luminosidad, s = size, color='blue', label='Luminosidad')
    plt.scatter(radio,masa, s = size, color='green', label='Masa')
    # plt.scatter(radio,opa, s = size, color='lime', label='Opacidad')
    # plt.axvline(x=r_front, ymin=0, ymax=75, color='purple', ls=':', lw=2, label='Frontera')
    plt.axvline(x=r_front, color='red', ls='--', lw=2, label='Frontera')
    plt.title("Parámetros en función del radio", fontsize=15)
    plt.axis([min(radio), max(radio), -0, max(max(luminosidad),max(presion))+5])
    plt.xlabel('$r$', fontsize=15)
    plt.legend(loc=1)
    plt.annotate('Radio frontera = '+str(round(r_front,3)), xy=(r_front+0.25, max(max(luminosidad),max(presion))-5), fontsize=12, color='r')
    plt.grid(color='grey', linestyle='--', lw=0.5)
    plt.show()    

    # Plot normalizado
    fig = plt.figure(figsize=(15,9))
    fig.patch.set_facecolor('xkcd:white')
    ax = plt.gca()
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')   
    size = np.ones_like(radio)*30
    plt.scatter(rr,presion/np.max(presion), s = size, color='black', label='Presión')
    plt.scatter(rr,temperatura/np.max(temperatura), s = size, color='orange', label='Temperatura')
    plt.scatter(rr,luminosidad/np.max(luminosidad), s = size, color='blue', label='Luminosidad')
    plt.scatter(rr,masa/np.max(masa), s = size, color='green', label='Masa')
    # plt.scatter(rr,opa/np.max(opa), s = size, color='lime', label='Opacidad')
    # plt.axvline(x=r_front, ymin=0, ymax=75, color='purple', ls=':', lw=2, label='Frontera')
    plt.axvline(x=r_front/max(radio), color='red', ls='--', lw=2, label='Frontera')
    plt.axis([0, 1, 0, 1])
    plt.title("Parámetros normalizados en función del radio", fontsize=15)
    plt.xlabel('$r$', fontsize=15)
    plt.legend(loc=5)
    plt.annotate('Radio frontera = '+str(round(r_front,3)), xy=(r_front/max(radio)+0.05, 0.9), fontsize=12, color='r')
    # ax.set_facecolor('xkcd:white')
    ax.axvspan(0, r_front/np.max(radio), facecolor='powderblue', alpha=0.2)
    ax.axvspan(r_front/np.max(radio), np.max(rr), facecolor='gold', alpha=0.2)
    plt.grid(color='grey', linestyle='--', lw=0.5)
    plt.show()

# end = time.time()
# print("El tiemo de ejecución es: ", end-start)


#funcion_graficas(M_tot, X, Y, R_tot, L_tot, T_central)
# ¡FUNCIONA!