import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def plot_rosenbrock():
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
    cbar = fig.colorbar(CS)

    # Graficar el punto óptimo
    ax.plot(optimal_point[0], optimal_point[1], 'ro', label='Punto óptimo')

    # Configurar límites de los ejes
    ax.set_xlim([-1.5, 1.1])
    ax.set_ylim([-0.5, 2.5])

    # Configurar etiquetas y título
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Función de Rosenbrock con restricciones')
    ax.legend()

    # Mostrar el gráfico
    plt.show()

# Llamar a la función para graficar
plot_rosenbrock()
