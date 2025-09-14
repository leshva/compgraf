import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Настройки изображения
w = 600
h = 600

bg_color = (255, 255, 255)  # серый фон
vp = np.full([h, w, 3], bg_color, dtype='uint8')

# Параметры квадратов
square_size = 20
spacing = 80

num_cols = w // spacing
num_rows = h // spacing

offset_x = (w - (num_cols - 1) * spacing) // 2
offset_y = (h - (num_rows - 1) * spacing) // 2

# Параметры перевернутой параболы (вершина в центре)
a = -0.007  # отрицательный коэффициент для перевернутой параболы
vertex_x = w // 2
vertex_y = h // 2 + 300
cutoff_height = 10  # Высота, ниже которой всё исчезает

# Параметры случайного смещения квадратов
POSITION_NOISE = 20  # Максимальное смещение квадратов


# Функция для определения, находится ли точка снаружи параболы и выше уровня отсечения
def is_outside_parabola(x, y):
    """Проверяет, находится ли точка снаружи перевернутой параболы и выше уровня отсечения"""
    if y < cutoff_height:  # Если ниже уровня отсечения - не рисуем
        return False

    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return y <= parabola_y  # для перевернутой параболы снаружи - ниже кривой


# Функция для вычисления расстояния до параболы
def distance_to_parabola(x, y):
    """Вычисляет вертикальное расстояние от точки до параболы"""
    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return abs(y - parabola_y)


# Создаем идеальную сетку центров
ideal_centers = []
for row in range(num_rows):
    row_centers = []
    for col in range(num_cols):
        center_x = offset_x + col * spacing
        center_y = offset_y + row * spacing
        row_centers.append((center_x, center_y))
    ideal_centers.append(row_centers)

# Находим ближайшие к параболе квадраты в каждом ряду и колонке
closest_in_row = [-1] * num_rows  # индекс колонки ближайшего квадрата для каждого ряда
closest_in_col = [-1] * num_cols  # индекс ряда ближайшего квадрата для каждой колонки

# Для каждого ряда находим квадрат, ближайший к параболе
for row in range(num_rows):
    min_distance = float('inf')
    closest_col = -1
    for col in range(num_cols):
        center_x, center_y = ideal_centers[row][col]
        if is_outside_parabola(center_x, center_y):
            dist = distance_to_parabola(center_x, center_y)
            if dist < min_distance:
                min_distance = dist
                closest_col = col
    closest_in_row[row] = closest_col

# Для каждой колонки находим квадрат, ближайший к параболе
for col in range(num_cols):
    min_distance = float('inf')
    closest_row = -1
    for row in range(num_rows):
        center_x, center_y = ideal_centers[row][col]
        if is_outside_parabola(center_x, center_y):
            dist = distance_to_parabola(center_x, center_y)
            if dist < min_distance:
                min_distance = dist
                closest_row = row
    closest_in_col[col] = closest_row

# Создаем сетку центров со смещением (ближайшие к параболе остаются на месте)
grid_centers = []
for row in range(num_rows):
    row_centers = []
    for col in range(num_cols):
        center_x, center_y = ideal_centers[row][col]

        # Если это ближайший квадрат в ряду или колонке - оставляем на месте
        if col == closest_in_row[row] or row == closest_in_col[col]:
            # Оставляем на идеальной позиции
            row_centers.append((center_x, center_y))
        else:
            # Смещаем случайным образом
            new_x = center_x + random.randint(-POSITION_NOISE, POSITION_NOISE)
            new_y = center_y + random.randint(-POSITION_NOISE, POSITION_NOISE)
            row_centers.append((new_x, new_y))
    grid_centers.append(row_centers)

# Собираем центры квадратов и их цвета только снаружи параболы и выше уровня отсечения
centers_outside = []
colors_outside = []

for row in range(num_rows):
    for col in range(num_cols):
        center_x, center_y = grid_centers[row][col]

        if is_outside_parabola(center_x, center_y):
            # Генерируем случайный цвет для квадрата
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)

            centers_outside.append((center_x, center_y, row, col))
            colors_outside.append(color)

            # Рисуем квадрат
            start_x = int(center_x - square_size // 2)
            start_y = int(center_y - square_size // 2)
            end_x = int(start_x + square_size)
            end_y = int(start_y + square_size)

            if (0 <= start_x < w and 0 <= end_x <= w and
                    0 <= start_y < h and 0 <= end_y <= h and start_y >= cutoff_height):
                vp[start_y:end_y, start_x:end_x] = color


# Функция для рисования прямой линии между двумя точками
def draw_line(x0, y0, x1, y1, color, thickness=2):
    """Рисует прямую линию алгоритмом Брезенхема"""
    if y0 < cutoff_height or y1 < cutoff_height:
        return

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx_total = x1 - x0
    dy_total = y1 - y0

    d = 2 * dy_total - dx_total
    dy2 = 2 * dy_total
    dx2 = 2 * dx_total

    y_step = 1 if y0 < y1 else -1

    x = x0
    y = y0

    points = []
    while x <= x1:
        if (steep and x < cutoff_height) or (not steep and y < cutoff_height):
            break

        if steep:
            points.append((y, x))
        else:
            points.append((x, y))

        if d > 0:
            y += y_step
            d -= dx2
        d += dy2
        x += 1

    # Рисуем точки с учетом толщины
    for px, py in points:
        for i in range(-thickness // 2, thickness // 2 + 1):
            for j in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = int(px + i), int(py + j)
                if 0 <= nx < w and 0 <= ny < h and ny >= cutoff_height:
                    vp[ny, nx] = color


# Соединяем квадраты линиями в правильном порядке
# Горизонтальные соединения
for row in range(num_rows):
    for col in range(num_cols - 1):
        center1_x, center1_y = grid_centers[row][col]
        center2_x, center2_y = grid_centers[row][col + 1]

        if (is_outside_parabola(center1_x, center1_y) and
                is_outside_parabola(center2_x, center2_y)):

            # Находим цвет первого квадрата
            color = None
            for i, (cx, cy, r, c) in enumerate(centers_outside):
                if r == row and c == col:
                    color = colors_outside[i]
                    break

            if color:
                draw_line(center1_x, center1_y, center2_x, center2_y, color, thickness=2)

# Вертикальные соединения
for row in range(num_rows - 1):
    for col in range(num_cols):
        center1_x, center1_y = grid_centers[row][col]
        center2_x, center2_y = grid_centers[row + 1][col]

        if (is_outside_parabola(center1_x, center1_y) and
                is_outside_parabola(center2_x, center2_y)):

            # Находим цвет первого квадрата
            color = None
            for i, (cx, cy, r, c) in enumerate(centers_outside):
                if r == row and c == col:
                    color = colors_outside[i]
                    break

            if color:
                draw_line(center1_x, center1_y, center2_x, center2_y, color, thickness=2)


# Функция для нахождения ближайшего квадрата к точке
def find_nearest_square_color(x, y):
    """Находит цвет ближайшего квадрата к точке (x, y)"""
    min_dist = float('inf')
    nearest_color = (0, 0, 0)

    for (cx, cy, row, col), color in zip(centers_outside, colors_outside):
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_color = color

    return nearest_color


# Рисуем параболу с учетом уровня отсечения
parabola_resolution = 1  # шаг для сглаживания параболы

for x in range(0, w, parabola_resolution):
    y_parabola = int(a * (x - vertex_x) ** 2 + vertex_y)

    # Не рисуем параболу ниже уровня отсечения
    if y_parabola < cutoff_height:
        continue

    if 0 <= y_parabola < h:
        # Находим цвет ближайшего квадрата
        parabola_color = find_nearest_square_color(x, y_parabola)

        # Рисуем точку параболы
        for dy in range(-3, 4):  # делаем параболу толще
            for dx in range(-1, 2):
                ny = y_parabola + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and ny >= cutoff_height:
                    vp[ny, nx] = parabola_color

# Рисуем линию отсечения СЕРЫМ цветом
for x in range(w):
    for dy in range(-2, 3):
        ny = cutoff_height + dy
        if 0 <= ny < h:
            vp[ny, x] = (255, 255, 255)  # Серая линия отсечения

plt.figure(figsize=(12, 12))
plt.imshow(vp)
plt.axis('off')
plt.show()