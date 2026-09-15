import random

random.seed(23)

N = 200
VALUES = []

for _ in range(N):
    value = round(random.gauss(26, 5), 1)
    value = max(15.0, min(37.0, value))
    VALUES.append(value)

with open('data/values.txt', 'w') as file:
    for v in VALUES:
        file.write(f'{v}\n')

INTERVALS = [
    (200, 400, 32),
    (400, 600, 56),
    (600, 800, 120),
    (800, 1000, 104),
    (1000, 1200, 88),
]

with open('data/intervals.txt', 'w') as file:
    for begin, end, count in INTERVALS:
        file.write(f'{begin} {end} {count}\n')

print(f'Создан data/values.txt: {N} значений')
print(f'Создан data/intervals.txt: {len(INTERVALS)} интервалов')