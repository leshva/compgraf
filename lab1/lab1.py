import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Настройки изображения
w = 600
h = 600

bg_color = (150, 150, 150)  # серый фон
vp = np.full([h, w, 3], bg_color, dtype='uint8')

# Параметры квадратов
square_size = 20
spacing = 40

num_cols = w // spacing
num_rows = h // spacing

offset_x = (w - (num_cols - 1) * spacing) // 2
offset_y = (h - (num_rows - 1) * spacing) // 2

# Параметры перевернутой параболы (вершина в центре)
a = -0.007  # отрицательный коэффициент для перевернутой параболы
vertex_x = w // 2
vertex_y = h // 2 + 300
cutoff_height = 10  # Высота, ниже которой всё исчезает


# Функция для определения, находится ли точка снаружи параболы и выше уровня отсечения
def is_outside_parabola(x, y):
    """Проверяет, находится ли точка снаружи перевернутой параболы и выше уровня отсечения"""
    if y < cutoff_height:  # Если ниже уровня отсечения - не рисуем
        return False

    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return y <= parabola_y  # для перевернутой параболы снаружи - ниже кривой


# Собираем центры квадратов и их цвета только снаружи параболы и выше уровня отсечения
centers_outside = []
colors_outside = []

for row in range(num_rows):
    for col in range(num_cols):
        center_x = offset_x + col * spacing
        center_y = offset_y + row * spacing

        if is_outside_parabola(center_x, center_y):
            # Генерируем случайный цвет для квадрата
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)

            centers_outside.append((center_x, center_y))
            colors_outside.append(color)

            # Рисуем квадрат
            start_x = center_x - square_size // 2
            start_y = center_y - square_size // 2
            end_x = start_x + square_size
            end_y = start_y + square_size

            if (0 <= start_x < w and 0 <= end_x <= w and
                    0 <= start_y < h and 0 <= end_y <= h and start_y >= cutoff_height):
                vp[start_y:end_y, start_x:end_x] = color


# Функция для интерполяции цвета между двумя точками
def interpolate_color(color1, color2, t):
    """Интерполирует между двумя цветами"""
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)


# Функция для нахождения ближайшего квадрата к точке
def find_nearest_square_color(x, y):
    """Находит цвет ближайшего квадрата к точке (x, y)"""
    min_dist = float('inf')
    nearest_color = (0, 0, 0)

    for (cx, cy), color in zip(centers_outside, colors_outside):
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_color = color

    return nearest_color


# Функция алгоритма Брезенхема для рисования линии с интерполяцией цветов
def draw_line_bresenham_interpolated(x0, y0, x1, y1, color_start, color_end, thickness=2):
    """Рисует линию алгоритмом Брезенхема с интерполяцией цветов"""
    # Проверяем, не находятся ли точки ниже уровня отсечения
    if y0 < cutoff_height or y1 < cutoff_height:
        return []

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
        color_start, color_end = color_end, color_start

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
        # Проверяем, не вышли ли за уровень отсечения
        if (steep and x < cutoff_height) or (not steep and y < cutoff_height):
            break

        # Вычисляем прогресс для интерполяции цвета
        progress = (x - x0) / dx_total if dx_total > 0 else 0
        current_color = interpolate_color(color_start, color_end, progress)

        if steep:
            points.append((y, x, current_color))
        else:
            points.append((x, y, current_color))

        if d > 0:
            y += y_step
            d -= dx2
        d += dy2
        x += 1

    # Рисуем точки с учетом толщины
    for px, py, color in points:
        for i in range(-thickness // 2, thickness // 2 + 1):
            for j in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = px + i, py + j
                if 0 <= nx < w and 0 <= ny < h and ny >= cutoff_height:
                    vp[ny, nx] = color

    return points


# Горизонтальные линии сетки (только между соседними квадратами в строке)
for i in range(len(centers_outside)):
    x0, y0 = centers_outside[i]

    # Ищем правого соседа в той же строке
    for j in range(len(centers_outside)):
        if i != j:
            x1, y1 = centers_outside[j]
            # Проверяем, что это правый сосед в той же строке
            if abs(y1 - y0) < spacing / 2 and abs(x1 - x0 - spacing) < spacing / 2:
                color1 = colors_outside[i]
                color2 = colors_outside[j]
                draw_line_bresenham_interpolated(x0, y0, x1, y1, color1, color2, thickness=2)

# Вертикальные линии сетки (только между соседними квадратами в столбце)
for i in range(len(centers_outside)):
    x0, y0 = centers_outside[i]

    # Ищем нижнего соседа в том же столбце
    for j in range(len(centers_outside)):
        if i != j:
            x1, y1 = centers_outside[j]
            # Проверяем, что это нижний сосед в том же столбце
            if abs(x1 - x0) < spacing / 2 and abs(y1 - y0 - spacing) < spacing / 2:
                color1 = colors_outside[i]
                color2 = colors_outside[j]
                draw_line_bresenham_interpolated(x0, y0, x1, y1, color1, color2, thickness=2)

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
            vp[ny, x] = (150, 150, 150)  # Серая линия отсечения

plt.figure(figsize=(12, 12))
plt.imshow(vp)
plt.axis('off')
plt.show()

