import numpy as np
import matplotlib.pyplot as plt
import math
from settings import *


def is_outside_parabola(x, y):
    """Проверяет, находится ли точка снаружи перевернутой параболы и выше уровня отсечения"""
    if y < CUTOFF_HEIGHT:  # Если ниже уровня отсечения - не рисуем
        return False
    parabola_y = A * (x - VERTEX_X) ** 2 + VERTEX_Y
    return y <= parabola_y  # для перевернутой параболы снаружи - ниже кривой


def interpolate_color(color1, color2, t):
    """Функция интерполяции цвета с ограничением значений"""
    return tuple(min(255, max(0, int(color1[i] + (color2[i] - color1[i]) * t))) for i in range(3))


def draw_horizontal_line_bresenham(x1, x2, y, color1, color2, image):
    """Рисует горизонтальную линию алгоритмом Брезенхема с интерполяцией цвета"""
    if y < CUTOFF_HEIGHT or y >= HEIGHT:
        return

    # Убедимся, что x1 <= x2
    if x1 > x2:
        x1, x2 = x2, x1
        color1, color2 = color2, color1

    x1 = max(0, min(int(x1), WIDTH - 1))
    x2 = max(0, min(int(x2), WIDTH - 1))

    dx = x2 - x1
    if dx == 0:
        # Точка
        if 0 <= x1 < WIDTH:
            image[y, x1] = color1
        return

    # Интерполируем цвет по горизонтали
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx if dx > 0 else 0
        color = interpolate_color(color1, color2, t)
        if 0 <= x < WIDTH:
            image[y, x] = color


def fill_rectangle_bresenham(vertices, colors, image, use_interpolation=True):
    """Закрашивает прямоугольник методом Брезенхема с использованием горизонтальных линий"""
    if len(vertices) != 4:
        return

    # Находим диапазон Y координат
    min_y = max(CUTOFF_HEIGHT, min(int(v[1]) for v in vertices))
    max_y = min(HEIGHT - 1, max(int(v[1]) for v in vertices))

    # Создаем список активных ребер
    active_edges = []

    # Обрабатываем все 4 ребра прямоугольника
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        color1 = colors[i]
        color2 = colors[(i + 1) % 4]

        # Пропускаем горизонтальные ребра
        if abs(y1 - y2) < 1e-6:
            continue

        # Определяем направление ребра (от меньшего Y к большему)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            color1, color2 = color2, color1

        # Добавляем информацию о ребре
        dy = y2 - y1
        if dy > 0:
            dx = (x2 - x1) / dy
            dc = tuple((c2 - c1) / dy for c1, c2 in zip(color1, color2)) if use_interpolation else (0, 0, 0)
            active_edges.append({
                'x': x1,
                'y_start': int(y1),
                'y_end': int(y2),
                'dx': dx,
                'color': list(color1),
                'dc': list(dc) if use_interpolation else [0, 0, 0]
            })

    # Если нет активных ребер, выходим
    if not active_edges:
        return

    # Сортируем ребра по начальной Y координате
    active_edges.sort(key=lambda edge: edge['y_start'])

    # Текущие активные ребра
    current_edges = []

    # Сканируем по строкам
    for y in range(min_y, max_y + 1):
        # Добавляем новые активные ребра
        while active_edges and active_edges[0]['y_start'] <= y:
            edge = active_edges.pop(0)
            current_edges.append(edge)

        # Удаляем завершенные ребра
        current_edges = [edge for edge in current_edges if edge['y_end'] >= y]

        # Если нет активных ребер, пропускаем строку
        if len(current_edges) < 2:
            continue

        # Сортируем активные ребра по X координате
        current_edges.sort(key=lambda edge: edge['x'])

        # Для каждой пары ребер рисуем горизонтальную линию с интерполяцией цвета
        for i in range(0, len(current_edges) - 1, 2):
            if i + 1 < len(current_edges):
                edge_left = current_edges[i]
                edge_right = current_edges[i + 1]

                x_start = edge_left['x']
                x_end = edge_right['x']

                if x_end <= x_start:
                    continue

                if use_interpolation:
                    # Получаем текущие цвета для левого и правого ребер
                    color_left = tuple(min(255, max(0, int(c))) for c in edge_left['color'])
                    color_right = tuple(min(255, max(0, int(c))) for c in edge_right['color'])

                    # Рисуем горизонтальную линию с интерполяцией цвета
                    draw_horizontal_line_bresenham(x_start, x_end, y, color_left, color_right, image)
                else:
                    # Используем цвет левого ребра для всей линии
                    color_left = tuple(min(255, max(0, int(c))) for c in edge_left['color'])
                    draw_horizontal_line_bresenham(x_start, x_end, y, color_left, color_left, image)

        # Обновляем X координаты и цвета для следующей строки
        for edge in current_edges:
            edge['x'] += edge['dx']
            if use_interpolation:
                for j in range(3):
                    edge['color'][j] += edge['dc'][j]
                    edge['color'][j] = max(0, min(255, edge['color'][j]))


def draw_line_bresenham_white(x0, y0, x1, y1, image, thickness=3):
    """Рисует белую линию алгоритмом Брезенхема"""
    # Пропускаем линии ниже уровня отсечения
    if y0 < CUTOFF_HEIGHT or y1 < CUTOFF_HEIGHT:
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
    dy_total = abs(y1 - y0)

    error = 0
    y = y0
    y_step = 1 if y0 < y1 else -1

    white_color = (255, 255, 255)  # Белый цвет

    for x in range(int(x0), int(x1) + 1):
        if steep:
            current_x, current_y = y, x
        else:
            current_x, current_y = x, y

        if current_y < CUTOFF_HEIGHT:
            continue

        for dx_offset in range(-thickness // 2, thickness // 2 + 1):
            for dy_offset in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = int(current_x + dx_offset), int(current_y + dy_offset)
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and ny >= CUTOFF_HEIGHT:
                    if dx_offset * dx_offset + dy_offset * dy_offset <= (thickness // 2) ** 2:
                        image[ny, nx] = white_color

        error += dy_total
        if 2 * error >= dx_total:
            y += y_step
            error -= dx_total


# Функция для нахождения ближайшего квадрата к точке
def find_nearest_square_color(x, y):
    """Находит цвет ближайшего квадрата к точке (x, y)"""
    min_dist = float('inf')
    nearest_color = BG_COLOR

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            center_x, center_y = CENTERS[row][col]
            if is_outside_parabola(center_x, center_y):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_color = VERTEX_COLORS[(col, row)]

    return nearest_color


def main():
    # Создаем изображение
    image = np.full([HEIGHT, WIDTH, 3], BG_COLOR, dtype='uint8')

    use_interpolation = FILL_MODE == "interpolate"

    # Сначала закрашиваем все прямоугольники
    filled_count = 0
    for polygon in POLYGONS:
        # Проверяем, что все вершины прямоугольника находятся снаружи параболы
        all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in polygon['vertices'])

        if all_vertices_outside:
            fill_rectangle_bresenham(polygon['vertices'], polygon['colors'], image, use_interpolation)
            filled_count += 1

    print(f"Закрашено {filled_count} прямоугольников (метод Брезенхема)")

    # Затем рисуем БЕЛЫЕ линии поверх закрашенных прямоугольников
    for polygon in POLYGONS:
        vertices = polygon['vertices']

        # Проверяем, что все вершины прямоугольника находятся снаружи параболы
        all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)

        if all_vertices_outside:
            # Рисуем 4 белые стороны прямоугольника
            for i in range(4):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % 4]
                draw_line_bresenham_white(x1, y1, x2, y2, image, thickness=3)

    # Рисуем квадраты в вершинах поверх линий (только снаружи параболы)
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            center_x, center_y = CENTERS[row][col]

            if is_outside_parabola(center_x, center_y):
                color = VERTEX_COLORS[(col, row)]

                # Рисуем квадрат
                start_x = int(center_x - SQUARE_SIZE // 2)
                start_y = int(center_y - SQUARE_SIZE // 2)
                end_x = int(start_x + SQUARE_SIZE)
                end_y = int(start_y + SQUARE_SIZE)

                # Гарантируем, что квадрат находится в пределах изображения и выше уровня отсечения
                start_x = max(0, min(start_x, WIDTH - SQUARE_SIZE))
                start_y = max(CUTOFF_HEIGHT, min(start_y, HEIGHT - SQUARE_SIZE))
                end_x = start_x + SQUARE_SIZE
                end_y = start_y + SQUARE_SIZE

                if start_y < HEIGHT and end_y > CUTOFF_HEIGHT:
                    image[start_y:end_y, start_x:end_x] = color

    # Рисуем параболу с учетом уровня отсечения
    parabola_resolution = 1
    for x in range(0, WIDTH, parabola_resolution):
        y_parabola = int(A * (x - VERTEX_X) ** 2 + VERTEX_Y)

        # Не рисуем параболу ниже уровня отсечения
        if y_parabola < CUTOFF_HEIGHT:
            continue

        if 0 <= y_parabola < HEIGHT:
            # Находим цвет ближайшего квадрата
            parabola_color = find_nearest_square_color(x, y_parabola)

            # Рисуем точку параболы
            for dy in range(-3, 4):
                for dx in range(-1, 2):
                    ny = y_parabola + dy
                    nx = x + dx
                    if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and ny >= CUTOFF_HEIGHT:
                        image[ny, nx] = parabola_color

    # Рисуем линию отсечения серым цветом
    for x in range(WIDTH):
        for dy in range(-2, 3):
            ny = CUTOFF_HEIGHT + dy
            if 0 <= ny < HEIGHT:
                image[ny, x] = BG_COLOR

    # Отображаем результат
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Выводим информацию о параметрах
    print(f"Параметры сетки:")
    print(f"Квадратов по X: {NUM_COLS}")
    print(f"Квадратов по Y: {NUM_ROWS}")
    print(f"Размер полигона деформации: {POLYGON_SIZE} пикселей")
    print(f"Прямоугольников: {len(POLYGONS)}")
    print(f"Режим закрашивания: {FILL_MODE}")
    print(f"Закрашивание: {'все прямоугольники' if FILL_ALL_RECTANGLES else 'один случайный'}")


if __name__ == "__main__":
    main()