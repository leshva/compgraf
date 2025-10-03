import numpy as np
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

# Настройки изображения
w = 600
h = 600
bg_color = (150, 150, 150)  # серый фон

# ИЗМЕНЯЕМЫЕ ПАРАМЕТРЫ
num_cols = 10  # число квадратов по X
num_rows = 10  # число квадратов по Y
polygon_size = 20  # размер полигона (области деформации)

# РЕЖИМЫ РАБОТЫ
FILL_MODE = "texture"  # "solid" - одним цветом, "interpolate" - интерполяция цветов, "texture" - текстура
AFFINE_MODE = "all"  # "single" - один случайный полигон, "all" - вся сетка
APPLY_AFFINE = True  # Применять аффинные преобразования

# ПАРАМЕТРЫ АФФИННЫХ ПРЕОБРАЗОВАНИЙ
SCALE_FACTOR = 1
ROTATION_ANGLE = 0
TRANSLATION_X = 0
TRANSLATION_Y = 0

# НАСТРОЙКИ ТЕКСТУРЫ
TEXTURE_PATH = "images.jpg"  # Путь к файлу текстуры

# Автоматический расчет spacing на основе числа квадратов
spacing_x = w // (num_cols - 1) if num_cols > 1 else w
spacing_y = h // (num_rows - 1) if num_rows > 1 else h
spacing = min(spacing_x, spacing_y)

# Параметры квадратов
square_size = 25

offset_x = (w - (num_cols - 1) * spacing) // 2
offset_y = (h - (num_rows - 1) * spacing) // 2

# Параметры перевернутой параболы (вершина в центре)
a = -0.007
vertex_x = w // 2
vertex_y = h // 2 + 300
cutoff_height = 10


# Загрузка текстуры
def load_texture(path):
    """Загружает текстуру из файла или создает дефолтную"""
    try:
        texture = Image.open(path)
        texture = texture.convert('RGB')
        print(f"Текстура загружена: {path} ({texture.size[0]}x{texture.size[1]})")
        return np.array(texture)
    except:
        # Создаем дефолтную текстуру если файл не найден
        print(f"Текстура {path} не найдена, создаю дефолтную")
        default_texture = np.zeros((256, 256, 3), dtype=np.uint8)
        # Создаем градиентную текстуру
        for i in range(256):
            for j in range(256):
                default_texture[i, j] = [
                    (i + j) % 256,  # R
                    (i * 2) % 256,  # G
                    (j * 2) % 256  # B
                ]
        return default_texture


# Загружаем текстуру
texture = load_texture(TEXTURE_PATH)
texture_height, texture_width = texture.shape[:2]


# Функция для определения, находится ли точка снаружи параболы и выше уровня отсечения
def is_outside_parabola(x, y):
    if y < cutoff_height:
        return False
    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return y <= parabola_y


# Функция для вычисления расстояния от точки до параболы
def distance_to_parabola(x, y):
    """Вычисляет расстояние от точки до параболы"""
    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return abs(y - parabola_y)


# Функция для определения, находится ли точка близко к параболе
def is_near_parabola(x, y, threshold=30):
    """Проверяет, находится ли точка близко к параболе"""
    return distance_to_parabola(x, y) < threshold


# Функции аффинных преобразований
def create_translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def create_rotation_matrix(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def create_scaling_matrix(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def create_affine_matrix(scale=1.0, rotation=0, translate_x=0, translate_y=0, center_x=0, center_y=0):
    T1 = create_translation_matrix(-center_x, -center_y)
    S = create_scaling_matrix(scale)
    R = create_rotation_matrix(rotation)
    T2 = create_translation_matrix(center_x + translate_x, center_y + translate_y)
    M = T2 @ R @ S @ T1
    return M


def apply_affine_matrix(point, matrix):
    x, y = point
    point_homogeneous = np.array([x, y, 1])
    transformed_point = matrix @ point_homogeneous
    return transformed_point[0], transformed_point[1]


def apply_affine_to_point(x, y, center_x, center_y, scale=1.0, rotation=0, translate_x=0, translate_y=0):
    M = create_affine_matrix(scale, rotation, translate_x, translate_y, center_x, center_y)
    return apply_affine_matrix((x, y), M)


# Функция для получения цвета из текстуры по UV-координатам
def get_texture_color(u, v):
    """Получает цвет из текстуры по UV-координатам (0-1)"""
    x = int(u * (texture_width - 1))
    y = int(v * (texture_height - 1))
    x = max(0, min(x, texture_width - 1))
    y = max(0, min(y, texture_height - 1))
    return tuple(texture[y, x])


# Функция интерполяции цвета с ограничением значений
def interpolate_color(color1, color2, t):
    return tuple(min(255, max(0, int(color1[i] + (color2[i] - color1[i]) * t))) for i in range(3))


# Алгоритм Брезенхема для рисования горизонтальной линии с текстурой
def draw_horizontal_line_texture(x1, x2, y, u1, u2, v1, v2, image):
    """Рисует горизонтальную линию с интерполяцией текстуры"""
    if y < cutoff_height or y >= h:
        return

    # Убедимся, что x1 <= x2
    if x1 > x2:
        x1, x2 = x2, x1
        u1, u2 = u2, u1
        v1, v2 = v2, v1

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))

    dx = x2 - x1
    if dx == 0:
        # Точка
        if 0 <= x1 < w:
            color = get_texture_color(u1, v1)
            image[y, x1] = color
        return

    # Интерполируем UV-координаты и рисуем линию
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx if dx > 0 else 0
        u = u1 + (u2 - u1) * t
        v = v1 + (v2 - v1) * t
        color = get_texture_color(u, v)
        if 0 <= x < w:
            image[y, x] = color


# Функция для закрашивания прямоугольника с текстурой (упрощенная версия)
def fill_rectangle_texture(vertices, image):
    """Закрашивает прямоугольник с наложением текстуры используя существующие ребра"""
    if len(vertices) != 4:
        return

    # UV-координаты для вершин прямоугольника
    uv_coords = [
        (0.0, 0.0),  # top_left
        (1.0, 0.0),  # top_right
        (1.0, 1.0),  # bottom_right
        (0.0, 1.0)  # bottom_left
    ]

    # Находим диапазон Y координат
    min_y = max(cutoff_height, min(int(v[1]) for v in vertices))
    max_y = min(h - 1, max(int(v[1]) for v in vertices))

    # Создаем список активных ребер с UV-координатами
    active_edges = []

    # Обрабатываем все 4 ребра прямоугольника
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        u1, v1 = uv_coords[i]
        u2, v2 = uv_coords[(i + 1) % 4]

        # Пропускаем горизонтальные ребра
        if abs(y1 - y2) < 1e-6:
            continue

        # Определяем направление ребра (от меньшего Y к большему)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            u1, v1, u2, v2 = u2, v2, u1, v1

        # Добавляем информацию о ребре с UV
        dy = y2 - y1
        if dy > 0:
            dx = (x2 - x1) / dy
            du = (u2 - u1) / dy
            dv = (v2 - v1) / dy
            active_edges.append({
                'x': x1,
                'y_start': int(y1),
                'y_end': int(y2),
                'dx': dx,
                'u': u1,
                'v': v1,
                'du': du,
                'dv': dv
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

        # Для каждой пары ребер рисуем горизонтальную линию с текстурой
        for i in range(0, len(current_edges) - 1, 2):
            if i + 1 < len(current_edges):
                edge_left = current_edges[i]
                edge_right = current_edges[i + 1]

                x_start = edge_left['x']
                x_end = edge_right['x']

                if x_end <= x_start:
                    continue

                # Получаем UV-координаты для левого и правого ребер
                u_left = edge_left['u']
                v_left = edge_left['v']
                u_right = edge_right['u']
                v_right = edge_right['v']

                # Рисуем горизонтальную линию с интерполяцией текстуры
                draw_horizontal_line_texture(x_start, x_end, y, u_left, u_right, v_left, v_right, image)

        # Обновляем X координаты и UV-координаты для следующей строки
        for edge in current_edges:
            edge['x'] += edge['dx']
            edge['u'] += edge['du']
            edge['v'] += edge['dv']


# Функция для закрашивания прямоугольника (основная)
def fill_rectangle_bresenham(vertices, colors, image, use_interpolation=True):
    """Закрашивает прямоугольник в зависимости от режима"""
    if FILL_MODE == "texture":
        # Используем текстурирование
        fill_rectangle_texture(vertices, image)
        return

    # Старый код для solid/interpolate режимов
    if len(vertices) != 4:
        return

    if not use_interpolation:
        solid_color = colors[0]

    min_y = max(cutoff_height, min(int(v[1]) for v in vertices))
    max_y = min(h - 1, max(int(v[1]) for v in vertices))

    active_edges = []

    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        color1 = colors[i]
        color2 = colors[(i + 1) % 4]

        if abs(y1 - y2) < 1e-6:
            continue

        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            color1, color2 = color2, color1

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

    if not active_edges:
        return

    active_edges.sort(key=lambda edge: edge['y_start'])
    current_edges = []

    for y in range(min_y, max_y + 1):
        while active_edges and active_edges[0]['y_start'] <= y:
            edge = active_edges.pop(0)
            current_edges.append(edge)

        current_edges = [edge for edge in current_edges if edge['y_end'] >= y]

        if len(current_edges) < 2:
            continue

        current_edges.sort(key=lambda edge: edge['x'])

        for i in range(0, len(current_edges) - 1, 2):
            if i + 1 < len(current_edges):
                edge_left = current_edges[i]
                edge_right = current_edges[i + 1]

                x_start = edge_left['x']
                x_end = edge_right['x']

                if x_end <= x_start:
                    continue

                if use_interpolation:
                    color_left = tuple(min(255, max(0, int(c))) for c in edge_left['color'])
                    color_right = tuple(min(255, max(0, int(c))) for c in edge_right['color'])
                    draw_horizontal_line_bresenham(x_start, x_end, y, color_left, color_right, image)
                else:
                    draw_horizontal_line_bresenham(x_start, x_end, y, solid_color, solid_color, image)

        for edge in current_edges:
            edge['x'] += edge['dx']
            if use_interpolation:
                for j in range(3):
                    edge['color'][j] += edge['dc'][j]
                    edge['color'][j] = max(0, min(255, edge['color'][j]))


# Старая функция для рисования линий (для solid/interpolate режимов)
def draw_horizontal_line_bresenham(x1, x2, y, color1, color2, image):
    if y < cutoff_height or y >= h:
        return
    if x1 > x2:
        x1, x2 = x2, x1
        color1, color2 = color2, color1
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    dx = x2 - x1
    if dx == 0:
        if 0 <= x1 < w:
            image[y, x1] = color1
        return
    for x in range(x1, x2 + 1):
        t = (x - x1) / dx if dx > 0 else 0
        color = interpolate_color(color1, color2, t)
        if 0 <= x < w:
            image[y, x] = color


# Метод Брезенхема для рисования белой линии
def draw_line_bresenham_white(x0, y0, x1, y1, image, thickness=3):
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
    white_color = (255, 255, 255)
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


# Функция для рисования параболы
def draw_parabola(image, color=(0, 0, 0), thickness=2):
    """Рисует параболу на изображении"""
    for x in range(0, w):
        # Вычисляем y координату параболы
        y_parabola = int(a * (x - vertex_x) ** 2 + vertex_y)

        # Пропускаем точки ниже уровня отсечения
        if y_parabola < cutoff_height:
            continue

        # Рисуем толстую линию параболы
        for dy in range(-thickness, thickness + 1):
            for dx in range(-thickness, thickness + 1):
                ny = y_parabola + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and ny >= cutoff_height:
                    # Проверяем чтобы точка была близко к параболе (для гладкости)
                    if abs(dy) <= thickness and abs(dx) <= thickness:
                        image[ny, nx] = color


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


# Генерируем случайные цвета для вершин
vertex_colors = {}
for row in range(num_rows):
    for col in range(num_cols):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        vertex_colors[(col, row)] = (r, g, b)

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
        near_parabola = is_near_parabola(base_x, base_y)
        if near_parabola or row == 0 or row == num_rows - 1 or col == 0 or col == num_cols - 1:
            center_x = base_x
            center_y = base_y
        else:
            center_x = base_x + random.randint(-polygon_size, polygon_size)
            center_y = base_y + random.randint(-polygon_size, polygon_size)
        if APPLY_AFFINE and AFFINE_MODE == "all":
            center_x, center_y = apply_affine_to_point(
                center_x, center_y, grid_center_x, grid_center_y,
                SCALE_FACTOR, ROTATION_ANGLE, TRANSLATION_X, TRANSLATION_Y
            )
        row_centers.append((center_x, center_y))
    centers.append(row_centers)

# Создаем структуру данных для прямоугольников
polygons = []
valid_polygons = []
transformed_polygon = None

for row in range(num_rows - 1):
    for col in range(num_cols - 1):
        top_left = centers[row][col]
        top_right = centers[row][col + 1]
        bottom_left = centers[row + 1][col]
        bottom_right = centers[row + 1][col + 1]
        colors = [
            vertex_colors[(col, row)],
            vertex_colors[(col + 1, row)],
            vertex_colors[(col + 1, row + 1)],
            vertex_colors[(col, row + 1)]
        ]
        vertices = [top_left, top_right, bottom_right, bottom_left]
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
        all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)
        if all_vertices_outside:
            valid_polygons.append(polygon)

# ПРИМЕНЯЕМ АФФИННЫЕ ПРЕОБРАЗОВАНИЯ К ОДНОМУ СЛУЧАЙНОМУ ПОЛИГОНУ
if APPLY_AFFINE and AFFINE_MODE == "single" and valid_polygons:
    transformed_polygon = random.choice(valid_polygons)
    original_vertices = transformed_polygon['vertices']
    transformed_vertices = []
    for x, y in original_vertices:
        new_x, new_y = apply_affine_to_point(
            x, y, transformed_polygon['center'][0], transformed_polygon['center'][1],
            SCALE_FACTOR, ROTATION_ANGLE, TRANSLATION_X, TRANSLATION_Y
        )
        transformed_vertices.append((new_x, new_y))
    transformed_polygon['vertices'] = transformed_vertices
    print(f"Преобразован полигон с ID: {transformed_polygon['id']}")

# Создаем изображение
image = np.full([h, w, 3], bg_color, dtype='uint8')

use_interpolation = FILL_MODE == "interpolate"

# ЗАКРАШИВАЕМ В ЗАВИСИМОСТИ ОТ РЕЖИМА
filled_count = 0

if AFFINE_MODE == "all":
    for polygon in valid_polygons:
        fill_rectangle_bresenham(polygon['vertices'], polygon['colors'], image, use_interpolation)
        filled_count += 1
    print(f"Закрашены ВСЕ прямоугольники: {filled_count} шт")

elif AFFINE_MODE == "single" and transformed_polygon:
    fill_rectangle_bresenham(transformed_polygon['vertices'], transformed_polygon['colors'], image, use_interpolation)
    filled_count = 1
    print(f"Закрашен ТОЛЬКО преобразованный прямоугольник: ID {transformed_polygon['id']}")

# Затем рисуем БЕЛЫЕ линии поверх закрашенных прямоугольников
for polygon in polygons:
    vertices = polygon['vertices']
    all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)
    if all_vertices_outside:
        if AFFINE_MODE == "single" and transformed_polygon and polygon['id'] == transformed_polygon['id']:
            continue
        for i in range(4):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 4]
            draw_line_bresenham_white(x1, y1, x2, y2, image, thickness=3)

# Рисуем квадраты в вершинах
for row in range(num_rows):
    for col in range(num_cols):
        center_x, center_y = centers[row][col]
        if is_outside_parabola(center_x, center_y):
            color = vertex_colors[(col, row)]
            start_x = int(center_x - square_size // 2)
            start_y = int(center_y - square_size // 2)
            end_x = int(start_x + square_size)
            end_y = int(start_y + square_size)
            start_x = max(0, min(start_x, w - square_size))
            start_y = max(cutoff_height, min(start_y, h - square_size))
            end_x = start_x + square_size
            end_y = start_y + square_size
            if start_y < h and end_y > cutoff_height:
                image[start_y:end_y, start_x:end_x] = color

# РИСУЕМ ПАРАБОЛУ (красным цветом)
draw_parabola(image, color=(255, 255, 255), thickness=3)

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

title = f"Режим: {FILL_MODE}"
if FILL_MODE == "texture":
    title += " (текстура)"
if APPLY_AFFINE:
    if AFFINE_MODE == "single":
        title += f" | Аффины: ОДИН полигон"
    else:
        title += f" | Аффины: ВСЯ СЕТКА"
    title += f"\nМасштаб: {SCALE_FACTOR}, Поворот: {ROTATION_ANGLE}°, Сдвиг: ({TRANSLATION_X}, {TRANSLATION_Y})"
plt.title(title)

plt.show()

print(f"\n=== ПАРАМЕТРЫ ===")
print(f"Сетка: {num_cols}x{num_rows} квадратов")
print(f"Режим закрашивания: {FILL_MODE}")
if FILL_MODE == "texture":
    print(f"Текстура: {TEXTURE_PATH} ({texture_width}x{texture_height})")
print(f"Аффинные преобразования: {'ВКЛ' if APPLY_AFFINE else 'ВЫКЛ'}")
if APPLY_AFFINE:
    print(f"Режим преобразований: {'ОДИН случайный' if AFFINE_MODE == 'single' else 'ВСЯ СЕТКА'}")
print(f"Закрашено прямоугольников: {filled_count}")