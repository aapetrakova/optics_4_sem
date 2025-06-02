import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
import matplotlib.colors as colors

# Параметры системы
a = 1.0            # Период решетки
lambda_ = 1.0      # Длина волны
zT = 2 * a**2 / lambda_  # Длина Талбота

# Параметры решетки
slit_width = 0.2 * a  # Ширина щели
nslits = 20          # Количество щелей

# Область визуализации
x_ratio_max = 3.0    # Макс. x в единицах a
z_ratio_max = 3.0    # Макс. z в единицах zT
dx = 0.01            # Шаг по x
dz = 0.01            # Шаг по z

# Создаем координатную сетку
x_ratio = np.arange(-x_ratio_max, x_ratio_max, dx)  # x/a
z_ratio = np.arange(0.01, z_ratio_max, dz)          # z/zT
X, Z = np.meshgrid(x_ratio, z_ratio)

# Массив для интенсивности
intensity = np.zeros_like(X)

# Вычисляем интенсивность для каждой точки
for i in range(len(z_ratio)):
    z = z_ratio[i] * zT  # Абсолютное z
    
    # Масштабный коэффициент Френеля
    scale = np.sqrt(z * lambda_ / np.pi)
    
    # Нормированные координаты
    sx = X[i,:] * a / scale  # x/a -> безразмерный аргумент
    sw = slit_width / scale  # ширина щели
    sp = a / scale           # период решетки
    
    c_tot, s_tot = 0.0, 0.0
    
    for j in range(nslits):
        # Положение j-й щели
        pos = (j - nslits//2) * sp
        
        # Границы щели
        x1 = sw/2 - pos - sx
        x2 = -sw/2 - pos - sx
        
        # Интегралы Френеля
        c1, s1 = fresnel(x1)
        c2, s2 = fresnel(x2)
        
        c_tot += (c1 - c2)
        s_tot += (s1 - s2)
    
    intensity[i,:] = (c_tot**2 + s_tot**2)

# Создание кастомной цветовой карты
colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0.5, 0, 0)]
cmap = LinearSegmentedColormap.from_list('talbot', colors)

# Визуализация
plt.figure(figsize=(12, 6))
plt.imshow(intensity,
           extent=[-x_ratio_max, x_ratio_max, z_ratio_max, 0],
           cmap=cmap,
           aspect='auto')

plt.colorbar(label='Интенсивность')
plt.xlabel(r'$x/a$', fontsize=12)
plt.ylabel(r'$z/z_T$', fontsize=12)
plt.title('2D визуализация ковра Талбота', fontsize=14)

# Добавляем линии, отмечающие длины Талбота
for n in range(1, int(z_ratio_max)+1):
    plt.axhline(y=n, color='cyan', linestyle='--', alpha=0.5)
    plt.text(x_ratio_max*0.9, n-0.05, fr'z={n}$z_T$', 
             color='cyan', ha='right', va='top')

plt.grid(False)
plt.tight_layout()
plt.show()
