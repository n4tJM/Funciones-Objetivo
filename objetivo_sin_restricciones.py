import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def bird_mishra(x, y):
    return np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(x) * np.exp((1 - np.sin(y))**2) + (x - y)**2

def townsend(x, y):
    return (-(np.cos((x - 0.1) * y))**2) - (x * np.sin(3 * x + y)) 

def gomez_levy(x, y):
    return (4*x**2) - (2.1*x**4) + ((1/3)*x**6) + (x*y) - (4*y**2) + (4*y**4)

def simionescu(x, y):
    return 0.1 * (x * y)

def plot_rosenbrock_cubic_linear_constraint():
    # Generar una malla de puntos
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-0.5, 2.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # Definir las restricciones
    cubic_constraint = ((X - 1)**3) - Y + 1
    linear_constraint = X + Y - 2

    # Encontrar los puntos que satisfacen ambas restricciones
    inside_constraint = ((cubic_constraint <= 0)) & (linear_constraint <= 0)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point = [1.0, 1.0]

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contourf(X, Y, Z_masked, levels=np.logspace(-3, 3, 20), cmap='viridis', norm=mcolors.LogNorm())
    ax.contour(X, Y, Z_masked, levels=np.logspace(-8, 8, 20), colors='white', alpha = 0.6)
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar límites de los ejes
    ax.set_xlim([-1.5, 1.1])
    ax.set_ylim([-0.5, 2.5])

    # Configurar etiquetas y título
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función de Rosenbrock restringida con una cúbica y una recta')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

def plot_rosenbrock_disc_constraint():
    # Generar una malla de puntos
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # Definir las restricciones
    disc_constraint = (X**2) + (Y**2)

    # Encontrar los puntos que satisfacen ambas restricciones
    inside_constraint = (disc_constraint <= 2)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point = [1.0, 1.0]

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contourf(X, Y, Z_masked, levels=np.logspace(-3, 3, 20), cmap='viridis', norm=mcolors.LogNorm())
    ax.contour(X, Y, Z_masked, levels=np.logspace(-8, 8, 20), colors='white', alpha = 0.6)
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar límites de los ejes
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    # Configurar etiquetas y título
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función de Rosenbrock restringida a un disco')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

def plot_bird_mishra_with_constraints():
    # Generar una malla de puntos
    x = np.linspace(-10, 0, 400)
    y = np.linspace(-10, 0, 400)
    X, Y = np.meshgrid(x, y)
    Z = bird_mishra(X, Y)

    # Definir las restricciones
    constraint1 = (X + 5)**2 + (Y + 5)**2

    # Encontrar los puntos que satisfacen la restricción
    inside_constraint = (constraint1 < 25)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point = [-3.1302468, -1.5821422]

    # Analizar el rango de valores de Z
    Z_min, Z_max = Z.min(), Z.max()
    print(f"Rango de Z: {Z_min} a {Z_max}")

    # Definir los niveles de contorno
    levels = np.linspace(Z_min, Z_max, 20)

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))

     # Graficar contorno con líneas en blanco
    CS = ax.contourf(X, Y, Z_masked, levels=levels, cmap='viridis')
    ax.contour(X, Y, Z_masked, levels=levels, colors='white')
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar límites de los ejes
    ax.set_xlim([-10, 0])
    ax.set_ylim([-10, 0])

    # Configurar etiquetas y título
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función Bird de Mishra con restricciones')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

def plot_townsend_with_constraints():
    # Generar una malla de puntos
    x = np.linspace(-2.25, 2.25, 400)
    y = np.linspace(-2.5, 1.75, 400)
    X, Y = np.meshgrid(x, y)
    Z = townsend(X, Y)

    # Definir las restricciones
    t = np.arctan2(Y, X)
    constraint1 = X**2 + Y**2 
    constraint2 = ((2 * np.cos(t))- ((1/2)* np.cos(2*t))- ((1/4)* np.cos(3*t))- ((1/8)* np.cos(4*t)))**2 + (2 * np.sin(t))**2

    # Encontrar los puntos que satisfacen las restricciones
    inside_constraint = (constraint1 < constraint2)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point = [-2.0052938, 1.1944509]

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(np.min(Z), np.max(Z), 20)
    CS = ax.contourf(X, Y, Z_masked, levels=levels, cmap='viridis')
    ax.contour(X, Y, Z_masked, levels=levels, colors='white', alpha=0.4)

    # Añadir la barra de color
    cbar = fig.colorbar(CS)

    
    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar los límites de los ejes
    ax.set_xlim([-2.25, 2.25])
    ax.set_ylim([-2.5, 1.75])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función de Townsend modificada con restricciones')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

def plot_gomez_levy():
    # Generar una malla de puntos
    x = np.linspace(-1, 0.75, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = gomez_levy(X, Y)

    # Definir las restricciones
    constraint1 = -(np.sin(4 * np.pi * X)) + (2 * (np.sin(2 * np.pi * Y))**2)

    # Encontrar los puntos que satisfacen las restricciones
    inside_constraint = (constraint1 <= 1.5)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point = [0.08984201, -0.7126564]

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(np.min(Z), np.max(Z), 10)
    CS = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3)

    # Añadir la barra de color
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar los límites de los ejes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función de Gómez y Levy')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

def plot_simionescu():
    # Generar una malla de puntos
    x = np.linspace(-1.25, 1.25, 400)
    y = np.linspace(-1.25, 1.25, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # Definir las restricciones
    r_T = 1
    r_S = 0.2
    n = 8
    constraint1 = (X**2) + (Y**2)
    constraint2 = (r_T + r_S * np.cos(n * np.arctan(X/Y)))**2

    # Encontrar los puntos que satisfacen ambas restricciones
    inside_constraint = (constraint1 <= constraint2)

    # Enmascarar los valores fuera de las restricciones
    Z_masked = np.ma.masked_where(~inside_constraint, Z)

    # Punto óptimo conocido
    optimal_point1 = [0.84852813, -0.84852813]
    optimal_point2 = [-0.84852813, 0.84852813]

    # Crear el gráfico
    levels = np.linspace(np.min(Z), np.max(Z), 10)
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contourf(X, Y, Z_masked, levels=levels, cmap='viridis')
    ax.contour(X, Y, Z_masked, levels=levels, colors='white', alpha = 0.3)
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point1[0], optimal_point1[1], 'ro', label='Punto óptimo1')
    ax.plot(optimal_point2[0], optimal_point2[1], 'rx', label='Punto óptimo2')

    # Configurar límites de los ejes
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    # Configurar etiquetas y título
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función Simionescu')
    ax.legend()

    # Mostrar el gráfico
    plt.show()
