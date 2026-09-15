from math import sqrt
from math import log
import matplotlib.pyplot as plt
import random

INPUT_VALUES = 'data/values.txt'
INPUT_INTERVALS = 'data/intervals.txt'
OUTPUT_FILE = 'data/m.txt'
PLOT_FILE = 'docs/histogram.png'

def interval_calc(filename):
    y_data = []
    hist_data = []
    with open(filename, 'r') as file:
        for line in file:
            value = float(line.strip())
            y_data.append(value)
            hist_data.append(value)

    line_count = len(y_data)
    m = round(1 + 3.322 * log(line_count, 10))
    a = round((max(y_data) - min(y_data)) / m, 1)

    interval = {}
    with open(OUTPUT_FILE, 'w') as output_file:
        output_file.write('Интервалы\tКоличество\n')
        for n in range(m):
            begin = round(min(y_data) + n * a, 1)
            if n == m - 1:
                end = round(max(y_data), 1)
            else:
                end = round(begin + a, 1)
            interval[n] = []
            for t in y_data:
                if begin <= t <= end:
                    interval[n].append(t)
            output_file.write(f'{begin}-{end}\t{len(interval[n])}\n')

    sum_f = 0
    sum_xf = 0
    middle_interval = {}
    for n in range(m):
        begin = min(y_data) + n * a
        end = max(y_data) if n == m - 1 else begin + a
        middle_interval[n] = (begin + end) / 2
        count = len(interval[n])
        sum_f += count
        sum_xf += count * middle_interval[n]
    x_mean = sum_xf / sum_f

    d_mean = 0
    dispersion = 0
    for n in range(m):
        count = len(interval[n])
        d_mean += abs((middle_interval[n] - x_mean) * count) / sum_f
        dispersion += ((middle_interval[n] - x_mean) ** 2) * count / sum_f

    sigma = sqrt(dispersion)
    V = sigma * 100 / x_mean
    R = max(y_data) - min(y_data)
    V_R = R / x_mean
    V_d = d_mean / x_mean

    return m, V, hist_data, V_R, V_d


def interval_set(filename):
    y_data = []
    begin = []
    end = []
    hist_data = []

    with open(filename, 'r') as file:
        for line in file:
            elements = line.split()
            begin.append(float(elements[0]))
            end.append(float(elements[1]))
            y_data.append(int(elements[2]))

    line_count = len(begin)
    sum_f = 0
    sum_xf = 0
    middle_interval = {}
    for n in range(line_count):
        b = begin[n]
        e = end[n]
        count = y_data[n]
        middle_interval[n] = (b + e) / 2
        sum_f += count
        sum_xf += count * middle_interval[n]
        for _ in range(count):
            hist_data.append(random.uniform(b, e))
    x_mean = sum_xf / sum_f

    d_mean = 0
    dispersion = 0
    for n in range(line_count):
        count = y_data[n]
        d_mean += abs((middle_interval[n] - x_mean) * count) / sum_f
        dispersion += ((middle_interval[n] - x_mean) ** 2) * count / sum_f

    sigma = sqrt(dispersion)
    V = sigma * 100 / x_mean
    m = line_count
    R = max(end) - min(begin)
    V_R = R / x_mean
    V_d = d_mean / x_mean

    return m, V, hist_data, V_R, V_d


def print_results(name, m, V, V_R, V_d):
    print(f'Файл: {name}')
    print(f'Количество интервалов: {m}')
    print(f'Коэффициент вариации: {round(V, 2)}%')
    print(f'Коэффициент осцилляции: {round(V_R, 2)}%')
    print(f'Линейный коэффициент вариации: {round(V_d, 2)}%')
    if V >= 33:
        print('Вывод: выборка неоднородна')
    else:
        print('Вывод: выборка однородна')
    print('')


m1, V1, hist1, VR1, Vd1 = interval_calc(INPUT_VALUES)
print_results('values.txt (интервалы не заданы)', m1, V1, VR1, Vd1)

m2, V2, hist2, VR2, Vd2 = interval_set(INPUT_INTERVALS)
print_results('intervals.txt (интервалы заданы)', m2, V2, VR2, Vd2)

plt.figure(figsize=(10, 6))
plt.hist(hist1, bins=int(m1), color='#f1c40f', edgecolor='#c0392b', linewidth=1.2)
plt.title('Гистограмма данных (интервалы не заданы)')
plt.ylabel('Частота')
plt.xlabel('Значения')
plt.grid(alpha=0.3)
plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
plt.show()