import numpy as np

"""
    Resuelve una ecuación diferencial ordinaria usando el método de Euler.

    f: función que representa la ecuación diferencial, f(x, y)
    x0: valor inicial de x
    y0: valor inicial de y
    h: paso del método de Euler
    t: valor alrededor del cual queremos que se estabilice la solución
    tolerance: tolerancia aceptada para la estabilización
    max_iter: número máximo de iteraciones
    retorna dupla (x, y) con los puntos obtenidos

"""

def euler_method(f, x0, y0, h, t, tolerance=1e-6, max_iter=10000):
    
    x = x0
    y = y0
    i = 0

    while np.abs(y - t) > tolerance and i < max_iter:
        y = y + h * f(x, y)
        x = x + h
        i += 1

    if i == max_iter:
        print("Se alcanzó el número máximo de iteraciones sin estabilización.")

    return x, y

# Prueba del método de Euler con la función y' = y, y(0) = 1, buscando estabilización en y=2.
def f(x, y):
    return y

x0 = 0
y0 = 1
h = 0.01
t = 2

x, y = euler_method(f, x0, y0, h, t)
print(f"Solución en x={x}, y={y}")

"""
    Realiza un paso del método de Runge-Kutta de cuarto orden.

    f: función que representa el sistema de ecuaciones diferenciales, f(x, y)
    x: valor actual de x
    y: valor actual de y
    h: paso actual
    retorna el siguiente valor de y

"""

def rk4_step(f, x, y, h):
    
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

"""
    Resuelve un sistema de ecuaciones diferenciales usando el método de Runge-Kutta de cuarto orden.

    f: función que representa el sistema de ecuaciones diferenciales, f(x, y)
    x0: valor inicial de x
    y0: valor inicial de y
    h: paso inicial
    target: valor alrededor del cual queremos que se estabilice la solución
    tolerance: tolerancia aceptada para la estabilización
    max_iter: número máximo de iteraciones
    retorna dupla (x, y) con los puntos obtenidos

"""

def rk4(f, x0, y0, h, target, tolerance=1e-6, max_iter=10000):
    
    x = x0
    y = y0
    i = 0

    while np.linalg.norm(y - target) > tolerance and i < max_iter:
        y = rk4_step(f, x, y, h)
        x = x + h
        i += 1

    if i == max_iter:
        print("Se alcanzó el número máximo de iteraciones sin estabilización.")

    return x, y

# Prueba del método de Runge-Kutta con el sistema y' = y, y(0) = [1, 1], buscando estabilización en y=[2, 2].
def f(x, y):
    return y

x0 = 0
y0 = np.array([1, 1])
h = 0.01
target = np.array([2, 2])

x, y = rk4(f, x0, y0, h, target)
print(f"Solución en x={x}, y={y}")