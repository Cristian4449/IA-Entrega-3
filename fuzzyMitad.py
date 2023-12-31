import numpy as np
import skfuzzy as sk
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

EntradaPos=float(input("DIGITE LA POSICION   \n"))
EntradaPes=float(input("DIGITE EL PESO \n"))
EntradaV=float(input("DIGITE LA VELOCIDAD  \n"))

#Generar los universos de entrada
posicion_moto = ctrl.Antecedent(np.arange(0, 60, 1), 'Posicion Moto')
peso_moto = ctrl.Antecedent(np.arange(100, 700, 1), 'Peso Moto')
velocidad_moto = ctrl.Antecedent(np.arange(0, 400, 1), 'Velocidad Moto')

#Generar los universos de salida
fuerzafrenado_moto = ctrl.Consequent(np.arange(50,1000, 1), 'Fuerza Frenado')

#Generar funciones de pertenencia para posicion_moto
posicion_moto['cerca'] = sk.gauss2mf(posicion_moto.universe, 0, 10, 6, 6.5)
posicion_moto['media'] = sk.gaussmf(posicion_moto.universe, 30, 6)
posicion_moto['lejos'] = sk.gauss2mf(posicion_moto.universe, 58, 8, 62, 6.5)

posicion_moto.view()
plt.title("Funcion de pertenencia - Posicion Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()

#Generar funciones de pertenencia para peso_moto
peso_moto['liviano'] = sk.gauss2mf(peso_moto.universe, 100, 50, 200, 50)
peso_moto['mediano'] = sk.gaussmf(peso_moto.universe, 400, 60)
peso_moto['pesado'] = sk.gauss2mf(peso_moto.universe, 600, 50, 700, 100)

peso_moto.view()
plt.title("Funcion de pertenencia - Peso Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()

#Generar funciones de pertenencia para velocidad_moto
velocidad_moto['muy lento'] = sk.gauss2mf(velocidad_moto.universe, 1, 10, 20, 30)
velocidad_moto['lento'] = sk.gaussmf(velocidad_moto.universe, 120, 30)
velocidad_moto['media'] = sk.gaussmf(velocidad_moto.universe, 200, 30)
velocidad_moto['rapido'] = sk.gaussmf(velocidad_moto.universe, 280, 30)
velocidad_moto['muy rapido'] = sk.gauss2mf(velocidad_moto.universe, 350, 20, 400, 10)

velocidad_moto.view()
plt.title("Funcion de pertenencia - Velocidad Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()

#Generar funciones de pertenencia para fuerzafrenado_moto
fuerzafrenado_moto['debil'] = sk.gauss2mf(fuerzafrenado_moto.universe, 50, 10, 145, 100)
fuerzafrenado_moto['media'] = sk.gaussmf(fuerzafrenado_moto.universe, 500, 132)
fuerzafrenado_moto['fuerte'] = sk.gauss2mf(fuerzafrenado_moto.universe, 850, 100, 1000, 50)

fuerzafrenado_moto.view()
plt.title("Funcion de pertenencia - Fuerza frenado moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()


#REGLAS CON LA POSICIÓN CERCA
regla_1 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['liviano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_2 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['mediano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_3 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['pesado'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])

regla_4 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['liviano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_5 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['mediano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_6 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['pesado'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])

regla_10 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['liviano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_11 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['mediano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_12 = ctrl.Rule(posicion_moto['cerca'] & peso_moto['pesado'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])

#REGLAS CON LA POSICIÓN MEDIA
regla_16 = ctrl.Rule(posicion_moto['media'] & peso_moto['liviano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_17 = ctrl.Rule(posicion_moto['media'] & peso_moto['mediano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_18 = ctrl.Rule(posicion_moto['media'] & peso_moto['pesado'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])

regla_19 = ctrl.Rule(posicion_moto['media'] & peso_moto['liviano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_20 = ctrl.Rule(posicion_moto['media'] & peso_moto['mediano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_21 = ctrl.Rule(posicion_moto['media'] & peso_moto['pesado'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])

regla_25 = ctrl.Rule(posicion_moto['media'] & peso_moto['liviano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_26 = ctrl.Rule(posicion_moto['media'] & peso_moto['mediano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_27 = ctrl.Rule(posicion_moto['media'] & peso_moto['pesado'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])


#REGLAS CON LA POSICIÓN LEJOS
regla_31 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['liviano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_32 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['mediano'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])
regla_33 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['pesado'] & velocidad_moto['muy lento'], fuerzafrenado_moto['debil'])

regla_34 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['liviano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_35 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['mediano'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])
regla_36 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['pesado'] & velocidad_moto['lento'], fuerzafrenado_moto['debil'])

regla_40 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['liviano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_41 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['mediano'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])
regla_42 = ctrl.Rule(posicion_moto['lejos'] & peso_moto['pesado'] & velocidad_moto['rapido'], fuerzafrenado_moto['media'])

sistema_frenado = ctrl.ControlSystem([regla_1, regla_2, regla_3, regla_4, regla_5, regla_6, regla_10,
                                     regla_11, regla_12, regla_16, regla_17, regla_18, regla_19, regla_20,
                                     regla_21, regla_25, regla_26, regla_27,
                                     regla_31, regla_32, regla_33, regla_34, regla_35, regla_36, regla_40,
                                     regla_41, regla_42])

simulador_frenado = ctrl.ControlSystemSimulation(sistema_frenado, flush_after_run = 21*21+1)

#Asignacion de entradas


 
simulador_frenado.input['Posicion Moto'] = EntradaPos

simulador_frenado.input['Peso Moto'] = EntradaPes

simulador_frenado.input['Velocidad Moto'] = EntradaV

print('Posición: ')
for t in posicion_moto.terms:
    mval = np.interp(EntradaPos, posicion_moto.universe, posicion_moto[t].mf)
    print(t, mval)

print('\n\nPeso: ')
for t in peso_moto.terms:
    mval = np.interp(EntradaPes, peso_moto.universe, peso_moto[t].mf)
    print(t, mval)

print('\n\nVelocidad: ')
for t in velocidad_moto.terms:
    mval = np.interp(EntradaV, velocidad_moto.universe, velocidad_moto[t].mf)
    print(t, mval)

simulador_frenado.compute()

posicion_moto.view(sim=simulador_frenado)
plt.title("Funcion de pertenencia - Posicion Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()
peso_moto.view(sim=simulador_frenado)
plt.title("Funcion de pertenencia - Peso Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()
velocidad_moto.view(sim=simulador_frenado)
plt.title("Funcion de pertenencia - Velocidad Moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()

print('\n\nDebe aplicar una fuerza de ', simulador_frenado.output['Fuerza Frenado'], 'N')
fuerzafrenado_moto.view(sim=simulador_frenado)
plt.title("Funcion de pertenencia - Fuerza frenado moto")
plt.xlabel("Pertenencia")
plt.ylabel("Membresia")
plt.show()
