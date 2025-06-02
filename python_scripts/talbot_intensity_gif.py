import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from PIL import Image
import io

# Параметры решетки
a = 1.0          # ширина щели
off1 = 10.0      # расстояние между щелями
nslits = 48      # количество щелей
xmax = 20.0      # размер области наблюдения
dx = 0.1         # шаг по x
num_z = 500      # количество шагов по z
z_max = 2.0      # максимальное значение z

# Создаем координатную сетку
x = np.arange(-xmax, xmax, dx)
z = np.linspace(0.01, z_max, num_z)

# Функция для расчета интенсивности через интегралы Френеля с учетом alpha
def calculate_fresnel_intensity(alpha):
    intensity = np.zeros((num_z, len(x)))
    
    for i, zi in enumerate(z):
        # Модифицируем z с учетом alpha
        z_eff = zi / alpha if alpha != 0 else zi
        
        sa = a / np.sqrt(z_eff/np.pi)
        sx = x / np.sqrt(z_eff/np.pi)
        soff1 = off1 / np.sqrt(z_eff/np.pi)
        
        # Функция для расчета интегралов Френеля для jj-й щели
        def fresnel_for_slit(jj):
            c1, s1 = fresnel(sa/2 - (2*jj+1)*soff1 - sx)
            c2, s2 = fresnel(-sa/2 - (2*jj+1)*soff1 - sx)
            c3, s3 = fresnel(sa/2 + (2*jj+1)*soff1 - sx)
            c4, s4 = fresnel(-sa/2 + (2*jj+1)*soff1 - sx)
            return (c1-c2+c3-c4, s1-s2+s3-s4)
        
        # Суммируем вклады от всех щелей
        c_tot, s_tot = fresnel_for_slit(0)
        for jj in range(1, nslits):
            c, s = fresnel_for_slit(jj)
            c_tot += c
            s_tot += s
        
        intensity[i,:] = 0.5*(c_tot**2 + s_tot**2)
    
    return intensity

# Создаем фигуру
fig, ax = plt.subplots(figsize=(10, 6))
plt.xlabel('Поперечная координата x')
plt.ylabel('Продольная координата z')

# Начальное изображение
alpha_values = np.arange(0.1, 2.1, 0.1)
intensity = calculate_fresnel_intensity(alpha_values[0])
im = ax.imshow(intensity, extent=[-xmax, xmax, z[-1], z[0]], 
               cmap='inferno', aspect='auto', norm=colors.PowerNorm(gamma=0.3))
cbar = fig.colorbar(im, ax=ax, label='Интенсивность')
title = ax.set_title(f'α = {alpha_values[0]:.1f}')

# Функция обновления кадра
def update(frame):
    alpha = alpha_values[frame]
    intensity = calculate_fresnel_intensity(alpha)
    im.set_array(intensity)
    title.set_text(f'α = {alpha:.1f}')
    return im, title

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=len(alpha_values), interval=200, blit=False)

# Сохраняем как GIF
print("Создание анимации...")
gif_path = "animation.gif"
ani.save(gif_path, writer='pillow', fps=5, dpi=100)

print(f"Анимация сохранена как {gif_path}")
plt.close()
