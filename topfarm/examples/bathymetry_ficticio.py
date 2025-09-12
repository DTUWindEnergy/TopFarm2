import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def get_bathymetry_func(sigma, mu=0, x_peak_1=0, y_peak_1=1000, x_peak_2=3000, y_peak_2=300, RADIUS=1300.1):
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

    f = RegularGridInterpolator((x, y), g)
    return f


def plot_bathymetry(f, RADIUS=1300.1):
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
