from math import sqrt
from math import log
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

INPUT_FILE = 'data/base.txt'
OUTPUT_FILE = 'data/new_base.txt'
PLOT_FILE = 'docs/regression_plot.png'

x_data = []
y_data = []

with open(INPUT_FILE, 'r') as file:
    for line in file:
        elements = line.split()
        x_data.append(float(elements[0]))
        y_data.append(float(elements[1]))

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

print('Граница переменной x для избежания выбросов - [', round(x_lower, 3), round(x_upper, 3), ']')
print('Граница переменной y для избежания выбросов - [', round(y_lower, 3), round(y_upper, 3), ']')

x_trimmed = []
y_trimmed = []

for i in range(len(x_data)):
    if x_lower <= x_data[i] <= x_upper and y_lower <= y_data[i] <= y_upper:
        x_trimmed.append(x_data[i])
        y_trimmed.append(y_data[i])

with open(OUTPUT_FILE, 'w') as output_file:
    for i in range(len(x_trimmed)):
        output_file.write(str(x_trimmed[i]) + '\t' + str(y_trimmed[i]) + '\n')

data1 = []
data2 = []
x_sum = 0
y_sum = 0
sum_xy = 0
cov_xy = 0
square_x = 0
square_y = 0
round_number = 3
a_level = 0.05
p_level = 1 - a_level
line_count = sum(1 for line in open(OUTPUT_FILE))

with open(OUTPUT_FILE, 'r') as data:
    for i in range(line_count):
        x, y = data.readline().split()
        data1.append(float(x))
        data2.append(float(y))

x_mean_new = round(sum(data1) / line_count, 2)
y_mean_new = round(sum(data2) / line_count, 2)
n = line_count

with open(OUTPUT_FILE, 'r') as data:
    x_sko_new = 0
    y_sko_new = 0
    for u in range(line_count):
        x, y = data.readline().split()
        x = float(x)
        y = float(y)
        x_sum += x
        y_sum += y
        sum_xy += x * y
        square_x += x ** 2
        square_y += y ** 2
        cov_xy += (x - x_mean_new) * (y - y_mean_new)
        x_sko_new += (x - x_mean_new) ** 2
        y_sko_new += (y - y_mean_new) ** 2

b = (n * sum_xy - x_sum * y_sum) / (n * square_x - (x_sum ** 2))
a = y_mean_new - x_mean_new * b
a = round(a, round_number)
b = round(b, round_number)

print('')
print('Линейное уравнение:')
print('y = ', b, ' * x + ', a)

corr_r = round(cov_xy / sqrt(x_sko_new * y_sko_new), 2)
determination = corr_r ** 2
determination = round(determination, 2)

print('Коэффициент корреляции - ', corr_r)
print('Коэффициент детерминации - ', determination)

F_calc = (determination / (1 - determination)) * (line_count - 2)
F_calc = round(F_calc, 2)

n_X = len(data1)
group_int_number = lambda n: round(3.31 * log(n_X, 10) + 1) if round(3.31 * log(n_X, 10) + 1) >= 2 else 2
K_X = group_int_number(n_X)
dfn = K_X - 1
dfd = n_X - K_X
F_table = scipy.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)

print(f'Табличное значение статистики критерия Фишера: F_table = {round(F_table, 2)}')

if F_calc < F_table:
    print(f'Так как F_calc = {round(F_calc, 2)} < F_table = {round(F_table, 2)}, '
          'то на уровне значимости 0.05 признаётся статистическая незначимость уравнения регрессии в целом.')
else:
    print(f'Так как F_calc = {round(F_calc, 2)} >= F_table = {round(F_table, 2)}, '
          'то на уровне значимости 0.05 признаётся статистическая значимость уравнения в целом')

plt.figure(figsize=(10, 6))
plt.scatter(data1, data2, color='#c0392b', alpha=0.6, edgecolor='black', linewidth=0.3)
plt.title('График линейной регрессии')
plt.xlabel('Переменная x')
plt.ylabel('Переменная y')
x = np.linspace(min(data1), max(data1))
y = b * x + a
plt.grid(alpha=0.3)
plt.plot(x, y, color='#2980b9', linewidth=2)
plt.plot(x_mean_new, y_mean_new, 'D', color='#f39c12', markersize=8, markeredgecolor='black')
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
plt.show()