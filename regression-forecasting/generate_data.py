import random

random.seed(17)

N = 200
K = 0.168
B = 0.836
NOISE = 0.15

POINTS = []

for _ in range(N):
    x = random.uniform(1.2, 18.7)
    y = K * x + B + random.gauss(0, NOISE)
    POINTS.append((x, y))

with open('data/data.txt', 'w') as file:
    for x, y in POINTS:
        file.write(f'{round(x, 4)} {round(y, 4)}\n')

print(f'Создан файл data/data.txt с {len(POINTS)} парами значений')