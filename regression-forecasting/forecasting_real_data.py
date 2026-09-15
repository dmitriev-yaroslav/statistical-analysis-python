from math import sqrt
from math import log
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

PLOT_FILE = 'docs/regression_plot_real.png'

SIGN_FACTOR = 1.1
A_LEVEL = 0.05
P_LEVEL = 1 - A_LEVEL
DEC_PLACE = 3

data = sns.load_dataset('diamonds')
x_data = data['carat'].tolist()
y_data = data['price'].tolist()

n = len(x_data)

x_mean = sum(x_data) / n
y_mean = sum(y_data) / n

sum_square_x = 0
sum_square_y = 0
cov_xy = 0

for x, y in zip(x_data, y_data):
    sum_square_x += (x - x_mean) ** 2
    sum_square_y += (y - y_mean) ** 2
    cov_xy += (x - x_mean) * (y - y_mean)

sko_x = sqrt(sum_square_x / n)
sko_x2 = sko_x ** 2

a = cov_xy / sum_square_x
b = y_mean - x_mean * a
a = round(a, DEC_PLACE)
b = round(b, DEC_PLACE)

corr_r = round(cov_xy / sqrt(sum_square_x * sum_square_y), DEC_PLACE)
determination = round(corr_r ** 2, DEC_PLACE)

F_calc = round((determination / (1 - determination)) * (n - 2), 2)
K_X = round(3.31 * log(n, 10) + 1)
dfn = K_X - 1
dfd = n - K_X
F_table = scipy.stats.f.ppf(P_LEVEL, dfn, dfd, loc=0, scale=1)

print('Датасет: diamonds (бриллианты)')
print(f'Объём выборки: {n}')
print('')
print('Линейное уравнение:')
print('y = ', a, ' * x + ', b)
print('')
print('Коэффициент корреляции - ', corr_r)
print('Коэффициент детерминации - ', determination)
print('')
print(f'Табличное значение статистики критерия Фишера: F_table = {round(F_table, DEC_PLACE)}')

sqr_s_ost = sum((y - (a * x + b)) ** 2 for x, y in zip(x_data, y_data))
s_ost = sqr_s_ost / (n - 2)

x_forecast = SIGN_FACTOR * x_mean
y_forecast = a * x_forecast + b
error_forecast = sqrt(s_ost * (1 + 1 / n + ((x_forecast - x_mean) ** 2) / (n * sko_x2)))
t_crit = scipy.stats.t.ppf(1 - A_LEVEL / 2, n - 2)
delta = error_forecast * t_crit

print('')
print('Прогноз при значении признака-фактора', round(x_forecast, DEC_PLACE), ':', round(y_forecast, DEC_PLACE))
print('Ошибка прогноза:', round(error_forecast, DEC_PLACE))
print('Доверительный интервал:')
print(round(y_forecast - delta, DEC_PLACE), ' < y < ', round(y_forecast + delta, DEC_PLACE))

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='#c0392b', alpha=0.6, edgecolor='black', linewidth=0.3)
plt.title('Зависимость размера чаевых от суммы счёта')
plt.xlabel('Сумма счёта, доллары')
plt.ylabel('Чаевые, доллары')
x = np.linspace(min(x_data), max(x_data))
y = a * x + b
plt.grid(alpha=0.3)
plt.plot(x, y, color='#2980b9', linewidth=2)
plt.plot(x_mean, y_mean, 'D', color='#f39c12', markersize=8, markeredgecolor='black')
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
plt.show()