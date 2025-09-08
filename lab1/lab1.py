import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Настройки изображения
WIDTH: int = 600
HEIGHT: int = 600
BG_COLOR: tuple = (255, 255, 255)  # белый фон
canvas = np.full([HEIGHT, WIDTH, 3], BG_COLOR, dtype='uint8')


# Параметры квадратов
SQUARE_SIZE: int = 20
SPACING: int = 40  # кол-во квадратов

NUM_COLS: int = WIDTH // SPACING
NUM_ROWS: int = HEIGHT // SPACING

OFFSET_X: int = (WIDTH - (NUM_COLS - 1) * SPACING) // 2
OFFSET_Y: int = (HEIGHT - (NUM_ROWS - 1) * SPACING) // 2


# параметры перевернутой параболы (вершина в центре)
a: float = -0.007  # отрицательный коэф для перевернутой параболы
VERTEX_X: int = WIDTH // 2
VERTEX_Y: int = HEIGHT // 2 + 300
CUTOFF_HEIGHT: int = 10  # высота, ниже которой всё исчезает


def is_outside_parabola(x, y) -> bool:
    """Проверяет, находится ли точка снаружи перевернутой параболы и выше уровня отсечения"""
    if y < CUTOFF_HEIGHT:  # если ниже уровня отсечения - не рисуем
        return False

    parabola_y = a * (x - VERTEX_X) ** 2 + VERTEX_Y
    return y <= parabola_y  # для перевернутой параболы снаружи - ниже кривой


# Собираем центры квадратов и их цвета только снаружи параболы и выше уровня отсечения
centers_outside: list = []
colors_outside: list = []

for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        center_x = OFFSET_X + col * SPACING
        center_y = OFFSET_Y + row * SPACING

        if is_outside_parabola(center_x, center_y):
            # Генерируем случайный цвет для квадрата
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)

            centers_outside.append((center_x, center_y))
            colors_outside.append(color)

            # Рисуем квадрат
            start_x = center_x - SQUARE_SIZE // 2
            start_y = center_y - SQUARE_SIZE // 2
            end_x = start_x + SQUARE_SIZE
            end_y = start_y + SQUARE_SIZE

            if (0 <= start_x < WIDTH and 0 <= end_x <= WIDTH and
                    0 <= start_y < HEIGHT and 0 <= end_y <= HEIGHT and start_y >= CUTOFF_HEIGHT):
                canvas[start_y:end_y, start_x:end_x] = color


def interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    """Функция для интерполяции цвета между двумя точками, интерполирует между двумя цветами"""
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)


def find_nearest_square_color(x: int, y: int) -> tuple:
    """Находит цвет ближайшего квадрата к точке (x, y)"""
    min_dist = float('inf')
    nearest_color = (0, 0, 0)

    for (cx, cy), color in zip(centers_outside, colors_outside):
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_color = color

    return nearest_color


def draw_line_bresenham_interpolated(x0: int, y0: int, x1: int, y1: int, color_start: tuple, color_end: tuple, thickness=2):
    """Рисует линию алгоритмом Брезенхема с интерполяцией цветов"""
    if y0 < CUTOFF_HEIGHT or y1 < CUTOFF_HEIGHT:  # check points
        return []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx

    if steep:  # если крутизна, то меняем
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    if x0 > x1:  # приводим к виду, чтобы x0 < x1
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
        if (steep and x < CUTOFF_HEIGHT) or (not steep and y < CUTOFF_HEIGHT):
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
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and ny >= CUTOFF_HEIGHT:
                    canvas[ny, nx] = color

    return points


# горизонт. линии сетки (только между соседними квадратами в строке)
for i in range(len(centers_outside)):
    x0, y0 = centers_outside[i]

    # ищем правого соседа в той же строке
    for j in range(len(centers_outside)):
        if i != j:
            x1, y1 = centers_outside[j]
            # проверяем, что это правый сосед в той же строке
            if abs(y1 - y0) < SPACING / 2 and abs(x1 - x0 - SPACING) < SPACING / 2:
                color1 = colors_outside[i]
                color2 = colors_outside[j]
                draw_line_bresenham_interpolated(x0, y0, x1, y1, color1, color2, thickness=2)

# вертикал. линии сетки (только между соседними квадратами в столбце)
for i in range(len(centers_outside)):
    x0, y0 = centers_outside[i]

    # ищем нижнего соседа в том же столбце
    for j in range(len(centers_outside)):
        if i != j:
            x1, y1 = centers_outside[j]
            # проверяем, что это нижний сосед в том же столбце
            if abs(x1 - x0) < SPACING / 2 and abs(y1 - y0 - SPACING) < SPACING / 2:
                color1 = colors_outside[i]
                color2 = colors_outside[j]
                draw_line_bresenham_interpolated(x0, y0, x1, y1, color1, color2, thickness=2)

# рисуем параболу с учетом уровня отсечения
parabola_resolution = 1  # шаг для сглаживания параболы (лестничный эффект)

for x in range(0, WIDTH, parabola_resolution):
    y_parabola = int(a * (x - VERTEX_X) ** 2 + VERTEX_Y)

    # параболу ниже уровня отсечения не рисуем
    if y_parabola < CUTOFF_HEIGHT:
        continue

    if 0 <= y_parabola < HEIGHT:
        # находим цвет ближайшего квадрата
        parabola_color = find_nearest_square_color(x, y_parabola)

        # рисуем точку параболы
        for dy in range(-3, 4):  # делаем параболу толще
            for dx in range(-1, 2):
                ny = y_parabola + dy
                nx = x + dx
                if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and ny >= CUTOFF_HEIGHT:
                    canvas[ny, nx] = parabola_color

# рисуем линию отсечения СЕРЫМ цветом
for x in range(WIDTH):
    for dy in range(-2, 3):
        ny = CUTOFF_HEIGHT + dy
        if 0 <= ny < HEIGHT:
            canvas[ny, x] = (150, 150, 150)  # Серая линия отсечения

plt.figure(figsize=(12, 12))
plt.imshow(canvas)
plt.axis('off')
plt.show()

