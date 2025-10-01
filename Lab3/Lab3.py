import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Настройки изображения
w = 600
h = 600
bg_color = (150, 150, 150)  # серый фон

# ИЗМЕНЯЕМЫЕ ПАРАМЕТРЫ
num_cols = 10  # число квадратов по X
num_rows = 10  # число квадратов по Y
polygon_size = 20  # размер полигона (области деформации)

# Параметры закрашивания
FILL_MODE = "solid"  # "solid" - одним цветом, "interpolate" - интерполяция цветов

# ПЕРЕКЛЮЧАТЕЛЬ АФФИННЫХ ПРЕОБРАЗОВАНИЙ
AFFINE_MODE = "single"  # "single" - один случайный полигон, "all" - вся сетка

# ПАРАМЕТРЫ АФФИННЫХ ПРЕОБРАЗОВАНИЙ
APPLY_AFFINE = True  # Применять аффинные преобразования
SCALE_FACTOR = 1.5  # Коэффициент масштабирования
ROTATION_ANGLE = 40  # Угол поворота в градусах
TRANSLATION_X = 20  # Сдвиг по X
TRANSLATION_Y = 20  # Сдвиг по Y

# Автоматический расчет spacing на основе числа квадратов
spacing_x = w // (num_cols - 1) if num_cols > 1 else w
spacing_y = h // (num_rows - 1) if num_rows > 1 else h
spacing = min(spacing_x, spacing_y)

# Параметры квадратов
square_size = 25

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


# Функция для вычисления расстояния от точки до параболы
def distance_to_parabola(x, y):
    """Вычисляет расстояние от точки до параболы"""
    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return abs(y - parabola_y)


# Функция для определения, находится ли точка близко к параболе
def is_near_parabola(x, y, threshold=30):
    """Проверяет, находится ли точка близко к параболе"""
    return distance_to_parabola(x, y) < threshold


# МАТРИЦЫ АФФИННЫХ ПРЕОБРАЗОВАНИЙ
def create_translation_matrix(tx, ty):
    """Создает матрицу переноса"""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def create_rotation_matrix(angle_degrees):
    """Создает матрицу поворота"""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def create_scaling_matrix(sx, sy=None):
    """Создает матрицу масштабирования"""
    if sy is None:
        sy = sx
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def create_affine_matrix(scale=1.0, rotation=0, translate_x=0, translate_y=0, center_x=0, center_y=0):
    """Создает комбинированную матрицу аффинных преобразований"""
    # Матрица переноса в начало координат
    T1 = create_translation_matrix(-center_x, -center_y)

    # Матрица масштабирования
    S = create_scaling_matrix(scale)

    # Матрица поворота
    R = create_rotation_matrix(rotation)

    # Матрица переноса обратно + дополнительный сдвиг
    T2 = create_translation_matrix(center_x + translate_x, center_y + translate_y)

    # Комбинированная матрица: T2 * R * S * T1
    M = T2 @ R @ S @ T1
    return M


def apply_affine_matrix(point, matrix):
    """Применяет аффинную матрицу к точке"""
    x, y = point
    # Преобразуем точку в однородные координаты
    point_homogeneous = np.array([x, y, 1])
    # Применяем преобразование
    transformed_point = matrix @ point_homogeneous
    # Возвращаем обратно в декартовы координаты
    return transformed_point[0], transformed_point[1]


# Функции аффинных преобразований для всей сетки
def apply_affine_to_point(x, y, center_x, center_y, scale=1.0, rotation=0, translate_x=0, translate_y=0):
    """Применяет масштабирование, поворот и перенос к точке относительно центра"""
    # Создаем аффинную матрицу
    M = create_affine_matrix(scale, rotation, translate_x, translate_y, center_x, center_y)
    # Применяем матрицу к точке
    return apply_affine_matrix((x, y), M)


# Генерируем случайные цвета для вершин
vertex_colors = {}
for row in range(num_rows):
    for col in range(num_cols):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        vertex_colors[(col, row)] = (r, g, b)


# Функция интерполяции цвета с ограничением значений
def interpolate_color(color1, color2, t):
    return tuple(min(255, max(0, int(color1[i] + (color2[i] - color1[i]) * t))) for i in range(3))


# Алгоритм Брезенхема для рисования горизонтальной линии с интерполяцией цвета
def draw_horizontal_line_bresenham(x1, x2, y, color1, color2, image):
    """Рисует горизонтальную линию алгоритмом Брезенхема с интерполяцией цвета"""
    if y < cutoff_height or y >= h:
        return

    # Убедимся, что x1 <= x2
    if x1 > x2:
        x1, x2 = x2, x1
        color1, color2 = color2, color1

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))

    dx = x2 - x1
    if dx == 0:
        # Точка
        if 0 <= x1 < w:
            image[y, x1] = color1
        return

    # Интерполируем цвет по горизонтали
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx if dx > 0 else 0
        color = interpolate_color(color1, color2, t)
        if 0 <= x < w:
            image[y, x] = color


# Функция для закрашивания прямоугольника методом Брезенхема
def fill_rectangle_bresenham(vertices, colors, image, use_interpolation=True):
    """Закрашивает прямоугольник методом Брезенхема с использованием горизонтальных линий"""
    if len(vertices) != 4:
        return

    # В режиме solid используем только цвет первой вершины для всего прямоугольника
    if not use_interpolation:
        solid_color = colors[0]  # Берем цвет первой вершины

    # Находим диапазон Y координат
    min_y = max(cutoff_height, min(int(v[1]) for v in vertices))
    max_y = min(h - 1, max(int(v[1]) for v in vertices))

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
            # В режиме solid не используем интерполяцию цветов
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

        # Для каждой пары ребер рисуем горизонтальную линию
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
                    # В режиме solid используем один цвет для всей линии
                    draw_horizontal_line_bresenham(x_start, x_end, y, solid_color, solid_color, image)

        # Обновляем X координаты и цвета для следующей строки
        for edge in current_edges:
            edge['x'] += edge['dx']
            if use_interpolation:
                for j in range(3):
                    edge['color'][j] += edge['dc'][j]
                    edge['color'][j] = max(0, min(255, edge['color'][j]))


# Метод Брезенхема для рисования белой линии
def draw_line_bresenham_white(x0, y0, x1, y1, image, thickness=3):
    """Рисует белую линию алгоритмом Брезенхема"""
    # Пропускаем линии ниже уровня отсечения
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

        if current_y < cutoff_height:
            continue

        for dx_offset in range(-thickness // 2, thickness // 2 + 1):
            for dy_offset in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = int(current_x + dx_offset), int(current_y + dy_offset)
                if 0 <= nx < w and 0 <= ny < h and ny >= cutoff_height:
                    if dx_offset * dx_offset + dy_offset * dy_offset <= (thickness // 2) ** 2:
                        image[ny, nx] = white_color

        error += dy_total
        if 2 * error >= dx_total:
            y += y_step
            error -= dx_total


# Центр преобразования для всей сетки
grid_center_x = w // 2
grid_center_y = h // 2

# Создаем массив центров с деформацией
centers = []
for row in range(num_rows):
    row_centers = []
    for col in range(num_cols):
        base_x = offset_x + col * spacing
        base_y = offset_y + row * spacing

        # Проверяем, находится ли точка близко к параболе
        near_parabola = is_near_parabola(base_x, base_y)

        # НЕ деформируем точки около параболы и на границах
        if near_parabola or row == 0 or row == num_rows - 1 or col == 0 or col == num_cols - 1:
            center_x = base_x
            center_y = base_y
        else:
            # Деформируем только точки, которые далеко от параболы
            center_x = base_x + random.randint(-polygon_size, polygon_size)
            center_y = base_y + random.randint(-polygon_size, polygon_size)

        # ПРИМЕНЯЕМ АФФИННЫЕ ПРЕОБРАЗОВАНИЯ КО ВСЕЙ СЕТКЕ
        if APPLY_AFFINE and AFFINE_MODE == "all":
            center_x, center_y = apply_affine_to_point(
                center_x, center_y,
                grid_center_x, grid_center_y,
                SCALE_FACTOR, ROTATION_ANGLE, TRANSLATION_X, TRANSLATION_Y
            )

        row_centers.append((center_x, center_y))
    centers.append(row_centers)

# Создаем структуру данных для прямоугольников
polygons = []
valid_polygons = []
transformed_polygon = None  # Для хранения преобразованного полигона

for row in range(num_rows - 1):
    for col in range(num_cols - 1):
        top_left = centers[row][col]
        top_right = centers[row][col + 1]
        bottom_left = centers[row + 1][col]
        bottom_right = centers[row + 1][col + 1]

        # Цвета вершин прямоугольника
        colors = [
            vertex_colors[(col, row)],  # top_left (первая вершина)
            vertex_colors[(col + 1, row)],  # top_right
            vertex_colors[(col + 1, row + 1)],  # bottom_right
            vertex_colors[(col, row + 1)]  # bottom_left
        ]

        vertices = [top_left, top_right, bottom_right, bottom_left]

        # Сохраняем информацию о прямоугольнике
        polygon = {
            'id': (col, row),
            'vertices': vertices,
            'colors': colors,
            'center': (
                (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4,
                (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
            )
        }

        polygons.append(polygon)

        # Сохраняем только те прямоугольники, которые снаружи параболы
        all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)
        if all_vertices_outside:
            valid_polygons.append(polygon)

# ПРИМЕНЯЕМ АФФИННЫЕ ПРЕОБРАЗОВАНИЯ К ОДНОМУ СЛУЧАЙНОМУ ПОЛИГОНУ
if APPLY_AFFINE and AFFINE_MODE == "single" and valid_polygons:
    # Выбираем случайный полигон для преобразования
    transformed_polygon = random.choice(valid_polygons)

    # Сохраняем оригинальные вершины
    original_vertices = transformed_polygon['vertices']

    # Применяем преобразования
    transformed_vertices = []
    for x, y in original_vertices:
        new_x, new_y = apply_affine_to_point(
            x, y, transformed_polygon['center'][0], transformed_polygon['center'][1],
            SCALE_FACTOR, ROTATION_ANGLE, TRANSLATION_X, TRANSLATION_Y
        )
        transformed_vertices.append((new_x, new_y))

    # Обновляем вершины преобразованного полигона
    transformed_polygon['vertices'] = transformed_vertices

    print(f"Преобразован полигон с ID: {transformed_polygon['id']}")

# Создаем изображение
image = np.full([h, w, 3], bg_color, dtype='uint8')

use_interpolation = FILL_MODE == "interpolate"

# ЗАКРАШИВАЕМ В ЗАВИСИМОСТИ ОТ РЕЖИМА
filled_count = 0

if AFFINE_MODE == "all":
    # Закрашиваем все подходящие прямоугольники
    for polygon in valid_polygons:
        fill_rectangle_bresenham(polygon['vertices'], polygon['colors'], image, use_interpolation)
        filled_count += 1
    print(f"Закрашены ВСЕ прямоугольники: {filled_count} шт")

elif AFFINE_MODE == "single" and transformed_polygon:
    # Закрашиваем ТОЛЬКО преобразованный полигон
    fill_rectangle_bresenham(transformed_polygon['vertices'], transformed_polygon['colors'], image, use_interpolation)
    filled_count = 1
    print(f"Закрашен ТОЛЬКО преобразованный прямоугольник: ID {transformed_polygon['id']}")

# Затем рисуем БЕЛЫЕ линии поверх закрашенных прямоугольников
for polygon in polygons:
    vertices = polygon['vertices']

    # Проверяем, что все вершины прямоугольника находятся снаружи параболы
    all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)

    if all_vertices_outside:
        # Пропускаем рисование ребер для преобразованного полигона в режиме single
        if AFFINE_MODE == "single" and transformed_polygon and polygon['id'] == transformed_polygon['id']:
            continue  # Не рисуем белые линии для преобразованного полигона

        # Рисуем 4 белые стороны прямоугольника
        for i in range(4):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 4]
            draw_line_bresenham_white(x1, y1, x2, y2, image, thickness=3)

# Рисуем квадраты в вершинах поверх линий (только снаружи параболы)
for row in range(num_rows):
    for col in range(num_cols):
        center_x, center_y = centers[row][col]

        if is_outside_parabola(center_x, center_y):
            color = vertex_colors[(col, row)]

            # Рисуем квадрат
            start_x = int(center_x - square_size // 2)
            start_y = int(center_y - square_size // 2)
            end_x = int(start_x + square_size)
            end_y = int(start_y + square_size)

            # Гарантируем, что квадрат находится в пределах изображения и выше уровня отсечения
            start_x = max(0, min(start_x, w - square_size))
            start_y = max(cutoff_height, min(start_y, h - square_size))
            end_x = start_x + square_size
            end_y = start_y + square_size

            if start_y < h and end_y > cutoff_height:
                image[start_y:end_y, start_x:end_x] = color


# Функция для нахождения ближайшего квадрата к точке
def find_nearest_square_color(x, y):
    """Находит цвет ближайшего квадрата к точке (x, y)"""
    min_dist = float('inf')
    nearest_color = bg_color

    for row in range(num_rows):
        for col in range(num_cols):
            center_x, center_y = centers[row][col]
            if is_outside_parabola(center_x, center_y):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_color = vertex_colors[(col, row)]

    return nearest_color


# Рисуем параболу с учетом уровня отсечения
parabola_resolution = 1
for x in range(0, w, parabola_resolution):
    y_parabola = int(a * (x - vertex_x) ** 2 + vertex_y)

    # Не рисуем параболу ниже уровня отсечения
    if y_parabola < cutoff_height:
        continue

    if 0 <= y_parabola < h:
        # Находим цвет ближайшего квадрата
        parabola_color = find_nearest_square_color(x, y_parabola)

        # Рисуем точку параболы
        for dy in range(-3, 4):
            for dx in range(-1, 2):
                ny = y_parabola + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and ny >= cutoff_height:
                    image[ny, nx] = parabola_color

# Рисуем линию отсечения серым цветом
for x in range(w):
    for dy in range(-2, 3):
        ny = cutoff_height + dy
        if 0 <= ny < h:
            image[ny, x] = bg_color

# Отображаем результат
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.tight_layout()

# Заголовок в зависимости от режима
if APPLY_AFFINE:
    if AFFINE_MODE == "single":
        plt.title(
            f'Аффинные преобразования: ОДИН случайный прямоугольник\nМасштаб: {SCALE_FACTOR}, Поворот: {ROTATION_ANGLE}°, Сдвиг: ({TRANSLATION_X}, {TRANSLATION_Y})')
    else:
        plt.title(
            f'Аффинные преобразования: ВСЯ СЕТКА\nМасштаб: {SCALE_FACTOR}, Поворот: {ROTATION_ANGLE}°, Сдвиг: ({TRANSLATION_X}, {TRANSLATION_Y})')
else:
    plt.title('Без аффинных преобразований')

plt.show()

# Выводим информацию о параметрах
print(f"\n=== ПАРАМЕТРЫ ===")
print(f"Сетка: {num_cols}x{num_rows} квадратов")
print(f"Прямоугольников: {len(polygons)}")
print(f"Режим закрашивания: {FILL_MODE}")
print(f"Аффинные преобразования: {'ВКЛ' if APPLY_AFFINE else 'ВЫКЛ'}")
if APPLY_AFFINE:
    print(f"Режим преобразований: {'ОДИН случайный' if AFFINE_MODE == 'single' else 'ВСЯ СЕТКА'}")
    print(f"Масштабирование: {SCALE_FACTOR}")
    print(f"Поворот: {ROTATION_ANGLE}°")
    print(f"Перенос: X={TRANSLATION_X}, Y={TRANSLATION_Y}")
print(f"Закрашено прямоугольников: {filled_count}")