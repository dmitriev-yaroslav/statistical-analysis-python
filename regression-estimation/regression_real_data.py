from math import sqrt
from math import log
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

PLOT_FILE = 'docs/regression_plot_real.png'

data = sns.load_dataset('tips')

x_data = data['total_bill'].tolist()
y_data = data['tip'].tolist()

x_mean = sum(x_data) / len(x_data)
y_mean = sum(y_data) / len(y_data)

x_sko = 0
y_sko = 0
for x in x_data:
    x_sko += (x - x_mean) ** 2
for y in y_data:
    y_sko += (y - y_mean) ** 2

x_std = sqrt(x_sko / (len(x_data) - 1))
y_std = sqrt(y_sko / (len(y_data) - 1))

threshold = 1.5
x_lower = x_mean - threshold * x_std
x_upper = x_mean + threshold * x_std
y_lower = y_mean - threshold * y_std
y_upper = y_mean + threshold * y_std

x_trimmed = []
y_trimmed = []
for i in range(len(x_data)):
    if x_lower <= x_data[i] <= x_upper and y_lower <= y_data[i] <= y_upper:
        x_trimmed.append(x_data[i])
        y_trimmed.append(y_data[i])

data1 = x_trimmed
data2 = y_trimmed
n = len(data1)

x_sum = sum(data1)
y_sum = sum(data2)
sum_xy = sum(x * y for x, y in zip(data1, data2))
square_x = sum(x ** 2 for x in data1)
cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(data1, data2))
x_sko_new = sum((x - x_mean) ** 2 for x in data1)
y_sko_new = sum((y - y_mean) ** 2 for y in data2)

b = (n * sum_xy - x_sum * y_sum) / (n * square_x - (x_sum ** 2))
a = y_mean - x_mean * b

print('Набор данных: tips (чаевые в ресторане)')
print(f'Исходных наблюдений: {len(x_data)}')
print(f'После удаления выбросов: {n}')
print('')
print('Линейное уравнение:')
print(f'y = {round(b, 4)} * x + {round(a, 4)}')

corr_r = round(cov_xy / sqrt(x_sko_new * y_sko_new), 2)
determination = round(corr_r ** 2, 2)

print('Коэффициент корреляции - ', corr_r)
print('Коэффициент детерминации - ', determination)

F_calc = round((determination / (1 - determination)) * (n - 2), 2)

n_X = len(data1)
K_X = round(3.31 * log(n_X, 10) + 1)
dfn = K_X - 1
dfd = n_X - K_X
F_table = scipy.stats.f.ppf(0.95, dfn, dfd, loc=0, scale=1)

print(f'Табличное значение статистики критерия Фишера: F_table = {round(F_table, 2)}')

if F_calc < F_table:
    print(f'F_calc = {F_calc} < F_table = {round(F_table, 2)}. Уравнение статистически незначимо.')
else:
    print(f'F_calc = {F_calc} >= F_table = {round(F_table, 2)}. Уравнение статистически значимо.')

plt.figure(figsize=(10, 6))
plt.scatter(data1, data2, color='#c0392b', alpha=0.6, edgecolor='black', linewidth=0.3)
plt.title('Зависимость размера чаевых от суммы счёта')
plt.xlabel('Сумма счёта, доллары')
plt.ylabel('Чаевые, доллары')
x = np.linspace(min(data1), max(data1))
y = b * x + a
plt.grid(alpha=0.3)
plt.plot(x, y, color='#2980b9', linewidth=2)
plt.plot(x_mean, y_mean, 'D', color='#f39c12', markersize=8, markeredgecolor='black')
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
plt.show()