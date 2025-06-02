import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Параметры системы
a = 1.0                  # Период решетки
lambda_s = 0.5           # Длина волны второй гармоники
w0 = 1.0                 # Радиус пучка в фокусе (на выходе из кристалла)

# Область визуализации
x_ratio_max = 3.0        # Макс. x в единицах a
z_ratio_max = 3.0        # Макс. z в единицах z_T
dx = 0.01                # Шаг по x
dz = 0.01                # Шаг по z

# Расчет нормированной длины Тальбота для SH
zT0 = 2 * a**2 / lambda_s  # базовая длина Тальбота без увеличения

# Сетка
x_ratio = np.arange(-x_ratio_max, x_ratio_max, dx)  # x/a
z_ratio = np.arange(0.01, z_ratio_max, dz)          # z/zT0
X, Z = np.meshgrid(x_ratio, z_ratio)

# Преобразуем в физические координаты
x = X * a
z = Z * zT0

# Расчет параметра увеличения и радиуса пучка в плоскости наблюдения
w_z = w0 * np.sqrt(1 + (2 * z / (w0**2))**2)

# Коэффициенты Фурье решетки (ограничим числом членов)
N = 20
b_n = np.ones(2 * N + 1)  # амплитуды, например, все по 1 (для прямоугольной решетки)
n_vals = np.arange(-N, N + 1).reshape(-1, 1, 1)

# Вычисляем фазовые члены (саморепродукция)
phase_z = np.exp(-1j * np.pi * lambda_s * (n_vals**2) * (w0**2) * z / (a**2 * w_z**2))
phase_x = np.exp(1j * 2 * np.pi * n_vals * w0**2 * x / (a * w_z**2))

# Сумма по n
field = np.sum(b_n.reshape(-1, 1, 1) * phase_z * phase_x, axis=0)

# Интенсивность
intensity = np.abs(field)**2

# Создание кастомной цветовой карты
colors_list = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0.5, 0, 0)]
cmap = LinearSegmentedColormap.from_list('talbot_nonlinear', colors_list)

# Визуализация
plt.figure(figsize=(12, 6))
plt.imshow(intensity,
           extent=[-x_ratio_max, x_ratio_max, z_ratio_max, 0],
           cmap=cmap,
           aspect='auto')

plt.colorbar(label='Интенсивность')
plt.xlabel(r'$x/a$', fontsize=12)
plt.ylabel(r'$z/z_T$', fontsize=12)
plt.title('2D визуализация нелинейного ковра Талбота (вторая гармоника)', fontsize=14)

# Добавляем линии кратных длин Тальбота
for m in range(1, int(z_ratio_max)+1):
    plt.axhline(y=m, color='cyan', linestyle='--', alpha=0.5)
    plt.text(x_ratio_max * 0.9, m - 0.05, fr'$z={m}z_T$', 
             color='cyan', ha='right', va='top')

plt.grid(False)
plt.tight_layout()
plt.show()
