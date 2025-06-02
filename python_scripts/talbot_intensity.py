import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
import matplotlib.colors as colors

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

# Значения alpha (доли длины Талбота)
alphas = [1.0, 0.5, 0.25, 0.125]
titles = ['α = 1 (полная длина Талбота)',
          'α = 1/2 (половина длины Талбота)',
          'α = 1/4 (четверть длины Талбота)',
          'α = 1/8 (одна восьмая длины Талбота)']

# Создаем фигуру с 4 subplots
plt.figure(figsize=(15, 10))

for i, alpha in enumerate(alphas):
    # Рассчитываем интенсивность
    intensity = calculate_fresnel_intensity(alpha)
    
    
    # Создаем subplot
    plt.subplot(2, 2, i+1)
    plt.imshow(intensity, extent=[-xmax, xmax, z[-1], z[0]], 
               cmap='inferno', aspect='auto', norm=colors.PowerNorm(gamma=0.3))
    plt.colorbar(label='Интенсивность')
    plt.xlabel('Поперечная координата x')
    plt.ylabel('Продольная координата z')
    plt.title(titles[i])

plt.tight_layout()
plt.show()
