import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def get_bathymetry_func_circle(sigma, mu=0, x_peak_1=0, y_peak_1=1000, x_peak_2=3000, y_peak_2=300, RADIUS=1300.1):
    x1, y1 = np.meshgrid(
        np.linspace(0 - x_peak_1, x_peak_1, 100),
        np.linspace(y_peak_1, 0 - y_peak_1, 100),
    )
    d1 = np.sqrt(x1 * x1 + y1 * y1)
    g1 = np.exp(-((d1 - mu) ** 2 / (2.0 * sigma**2)))
    x2, y2 = np.meshgrid(
        np.linspace(0 - x_peak_2, x_peak_2, 100),
        np.linspace(y_peak_2, 0 - y_peak_2, 100),
    )
    d2 = np.sqrt(x2 * x2 + y2 * y2)
    g2 = np.exp(-((d2 - mu) ** 2 / (2.0 * sigma**2)))
    g = 5 * (g1**1.9) - 7 * g2 - 50

    x = np.linspace(-2 * RADIUS, 2 * RADIUS, 100)
    y = np.linspace(-2 * RADIUS, 2 * RADIUS, 100)

    return RegularGridInterpolator((x, y), g)


def plot_bathymetry_circle(f, RADIUS=1300.1):
    x = np.linspace(-2 * RADIUS, 2 * RADIUS, 100)
    y = np.linspace(-2 * RADIUS, 2 * RADIUS, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))
    plt.imshow(
        Z,
        extent=(-2 * RADIUS, 2 * RADIUS, -2 * RADIUS, 2 * RADIUS),
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar()
    plt.title("Water depth function")
    plt.show()


def get_bathymetry_func_rect(g, x_min, x_max, y_min, y_max):
    x = np.linspace(x_min - 1000, x_max + 1000, 100)
    y = np.linspace(y_min - 1000, y_max + 1000, 100)
    return RegularGridInterpolator((x, y), g)


def plot_bathymetry_rect(g, x_min, x_max, y_min, y_max):
    plt.imshow(g, extent=(x_min - 1000, x_max + 1000, y_min - 1000, y_max + 1000), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('2D Gaussian Function')
    plt.show()


def gaussian_surface(sigma, mu=0, x_peak_1=1000, y_peak_1=-1000, x_peak_2=4000, y_peak_2=-8000,
                     x_min=0, x_max=6000, y_min=-10000, y_max=0):
    x1, y1 = np.meshgrid(np.linspace(x_min - x_peak_1, x_max - x_peak_1, 100), np.linspace(y_min - y_peak_1, y_max - y_peak_1, 100))
    d1 = np.sqrt(x1 * x1 + y1 * y1)
    g1 = np.exp(-((d1 - mu)**2 / (2.0 * sigma**2)))
    x2, y2 = np.meshgrid(np.linspace(x_min - x_peak_2, x_max - x_peak_2, 100), np.linspace(y_min - y_peak_2, y_max - y_peak_2, 100))
    d2 = np.sqrt(x2 * x2 + y2 * y2)
    g2 = np.exp(-((d2 - mu)**2 / (2.0 * sigma**2)))
    g = 5 * g1 - 8 * g2 - 30
    return g
