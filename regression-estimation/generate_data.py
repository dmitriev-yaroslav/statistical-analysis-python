import random

random.seed(42)

N = 300
K = 2.5
B = 10
NOISE = 8

POINTS = []

for _ in range(N):
    x = random.uniform(10, 80)
    y = K * x + B + random.gauss(0, NOISE)
    POINTS.append((x, y))

POINTS.append((1, 10000))
POINTS.append((10000, 1))
POINTS.append((10000, 10000))

with open('data/base.txt', 'w') as file:
    for x, y in POINTS:
        file.write(f'{x} {y}\n')

print(f'Создан файл data/base.txt с {len(POINTS)} парами значений')