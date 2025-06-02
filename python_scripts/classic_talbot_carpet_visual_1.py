import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Параметры системы
a = 0.01  # период решетки (мм)
wavelength = 0.0005  # длина волны (мм)
N = 20  # количество учитываемых гармоник

# Длина Талбота
z_t = 2 * a**2 / wavelength


# 2D визуализация ковра Талбота
x_2d = np.linspace(-2*a, 2*a, 500)
alpha_2d = np.linspace(0, 1, 500)
X, Alpha = np.meshgrid(x_2d, alpha_2d)

# Расчет поля для 2D случая
G = np.zeros_like(X, dtype=complex)
for n in range(-N, N+1):
    c_n = fourier_coeffs(n, a)
    G += c_n * np.exp(1j*2*np.pi*n*X/a) * np.exp(-1j*np.pi*n**2*Alpha)

I_2d = np.abs(G)**2

# Создание кастомной цветовой карты
colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0.5, 0, 0)]
cmap = LinearSegmentedColormap.from_list('talbot', colors)

plt.figure(figsize=(10, 6))
plt.imshow(I_2d, extent=[-2, 2, 0, 1], aspect='auto', cmap=cmap, origin='lower')
plt.colorbar(label='Интенсивность')
plt.xlabel('x/a')
plt.ylabel('α = z/z_t')
plt.title('2D визуализация ковра Талбота')
plt.show()
