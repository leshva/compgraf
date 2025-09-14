import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Настройки изображения
w = 600
h = 600
bg_color = (150, 150, 150)  # серый фон

# ИЗМЕНЯЕМЫЕ ПАРАМЕТРЫ
num_cols = 6  # число квадратов по X
num_rows = 6  # число квадратов по Y
polygon_size = 80  # размер полигона (области деформации)

# Параметры закрашивания (можно менять)
FILL_MODE = "interpolate"  # "solid" - одним цветом, "interpolate" - интерполяция цветов

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

# Z-буфер для невыпуклых полигонов
z_buffer = np.full((h, w), -float('inf'), dtype=float)


# Функция для определения, находится ли точка снаружи параболы и выше уровня отсечения
def is_outside_parabola(x, y):
    """Проверяет, находится ли точка снаружи перевернутой параболы и выше уровня отсечения"""
    if y < cutoff_height:  # Если ниже уровня отсечения - не рисуем
        return False
    parabola_y = a * (x - vertex_x) ** 2 + vertex_y
    return y <= parabola_y  # для перевернутой параболы снаружи - ниже кривой


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


# Функция для проверки выпуклости полигона
def is_convex(vertices):
    """Проверяет, является ли полигон выпуклым"""
    if len(vertices) < 3:
        return False

    n = len(vertices)
    sign = 0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        x3, y3 = vertices[(i + 2) % n]

        # Вычисляем векторное произведение
        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)

        if abs(cross_product) < 1e-10:  # Коллинеарные точки
            continue

        if sign == 0:
            sign = 1 if cross_product > 0 else -1
        else:
            current_sign = 1 if cross_product > 0 else -1
            if current_sign != sign:
                return False

    return True


# Функция для закрашивания ВЫПУКЛОГО полигона методом Брезенхема (разверстка)
def fill_convex_polygon_bresenham(vertices, colors, image, use_interpolation=True):
    """Закрашивает выпуклый полигон методом Брезенхема с разверсткой"""
    if len(vertices) != 4:
        return

    # Находим диапазон Y координат
    min_y = max(cutoff_height, min(int(v[1]) for v in vertices))
    max_y = min(h - 1, max(int(v[1]) for v in vertices))

    # Создаем список активных ребер
    active_edges = []

    # Обрабатываем все 4 ребра полигона
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
            # Исправлено: правильное вычисление шага цвета
            dc = tuple((c2 - c1) / dy for c1, c2 in zip(color1, color2)) if use_interpolation else (0, 0, 0)
            active_edges.append({
                'x': x1,
                'y_start': int(y1),
                'y_end': int(y2),
                'dx': dx,
                'color': list(color1),
                'dc': list(dc) if use_interpolation else [0, 0, 0],
                'current_y': int(y1)  # Добавляем текущую Y позицию для точной интерполяции
            })

    # Если нет активных ребер, выходим
    if not active_edges:
        return

    # Сортируем ребра по начальной Y координате
    active_edges.sort(key=lambda edge: edge['y_start'])

    # Текущие активные ребра
    current_edges = []
    current_y = min_y

    # Сканируем по строкам
    for y in range(min_y, max_y + 1):
        # Добавляем новые активные ребра
        while active_edges and active_edges[0]['y_start'] <= y:
            edge = active_edges.pop(0)
            # Исправлено: правильная инициализация цвета для текущей строки
            if use_interpolation:
                progress = y - edge['y_start']
                edge_color = [edge['color'][i] + edge['dc'][i] * progress for i in range(3)]
                edge['color'] = edge_color
            current_edges.append(edge)

        # Удаляем завершенные ребра
        current_edges = [edge for edge in current_edges if edge['y_end'] >= y]

        # Если нет активных ребер, пропускаем строку
        if len(current_edges) < 2:
            continue

        # Сортируем активные ребра по X координате
        current_edges.sort(key=lambda edge: edge['x'])

        # Закрашиваем между парами ребер
        for i in range(0, len(current_edges) - 1, 2):
            if i + 1 < len(current_edges):
                edge_left = current_edges[i]
                edge_right = current_edges[i + 1]

                x_start = int(edge_left['x'])
                x_end = int(edge_right['x'])

                if x_end <= x_start:
                    continue

                # Интерполируем цвет между левым и правым ребром
                for x in range(x_start, x_end + 1):
                    if 0 <= x < w:
                        if use_interpolation:
                            t = (x - x_start) / (x_end - x_start) if x_end != x_start else 0.5
                            # Исправлено: используем сохраненные цвета ребер
                            color_left = tuple(min(255, max(0, int(c))) for c in edge_left['color'])
                            color_right = tuple(min(255, max(0, int(c))) for c in edge_right['color'])
                            final_color = interpolate_color(color_left, color_right, t)
                        else:
                            final_color = tuple(min(255, max(0, int(c))) for c in edge_left['color'])
                        image[y, x] = final_color

        # Обновляем X координаты и цвета для следующей строки
        for edge in current_edges:
            edge['x'] += edge['dx']
            if use_interpolation:
                for j in range(3):
                    edge['color'][j] += edge['dc'][j]
                    # Ограничиваем значения цвета
                    edge['color'][j] = max(0, min(255, edge['color'][j]))


# Функция для закрашивания НЕВЫПУКЛОГО полигона с Z-буфером (без барицентрических координат)
def fill_concave_polygon_zbuffer(vertices, colors, image, use_interpolation=True):
    """Закрашивает невыпуклый полигон с использованием Z-буфера"""
    if len(vertices) != 4:
        return

    # Для невыпуклых полигонов используем один цвет (первую вершину) без интерполяции
    fill_color = colors[0]  # Цвет первой вершины

    # Триангулируем полигон на 2 треугольника
    triangles = [
        (vertices[0], vertices[1], vertices[2]),
        (vertices[0], vertices[2], vertices[3])
    ]

    for triangle in triangles:
        v0, v1, v2 = triangle
        x0, y0 = v0
        x1, y1 = v1
        x2, y2 = v2

        # Находим bounding box треугольника
        min_x = max(0, min(int(x0), int(x1), int(x2)))
        max_x = min(w - 1, max(int(x0), int(x1), int(x2)))
        min_y = max(cutoff_height, min(int(y0), int(y1), int(y2)))
        max_y = min(h - 1, max(int(y0), int(y1), int(y2)))

        # Вычисляем нормаль к плоскости треугольника
        vec1 = [x1 - x0, y1 - y0, 0]
        vec2 = [x2 - x0, y2 - y0, 0]
        normal = np.cross(vec1, vec2)

        if abs(normal[2]) < 1e-10:
            continue

        # Уравнение плоскости: Ax + By + Cz + D = 0
        A, B, C = normal
        D = -A * x0 - B * y0 - C * 0  # Z = 0 в вершинах

        # Закрашиваем треугольник
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if point_in_triangle(x, y, x0, y0, x1, y1, x2, y2):
                    # Вычисляем Z координату (простая аппроксимация)
                    if abs(C) > 1e-10:
                        z = (-A * x - B * y - D) / C
                    else:
                        z = 0

                    # Проверяем Z-буфер
                    if z > z_buffer[y, x]:
                        z_buffer[y, x] = z
                        image[y, x] = fill_color


# Функция для проверки, находится ли точка внутри треугольника
def point_in_triangle(x, y, x0, y0, x1, y1, x2, y2):
    """Проверяет, находится ли точка (x, y) внутри треугольника"""
    # Вычисляем площади треугольников
    area_total = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area1 = abs((x0 - x) * (y1 - y) - (x1 - x) * (y0 - y))
    area2 = abs((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y))
    area3 = abs((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y))

    # Точка внутри, если сумма площадей маленьких треугольников равна площади большого
    return abs(area1 + area2 + area3 - area_total) < 1e-6


# Метод Брезенхема для рисования линии
def draw_line_bresenham(x0, y0, x1, y1, color1, color2, image, thickness=3):
    """Рисует линию алгоритмом Брезенхема с интерполяцией цвета"""
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
        color1, color2 = color2, color1

    dx_total = x1 - x0
    dy_total = abs(y1 - y0)

    error = 0
    y = y0
    y_step = 1 if y0 < y1 else -1

    total_length = math.sqrt(dx_total ** 2 + dy_total ** 2)

    for x in range(int(x0), int(x1) + 1):
        if steep:
            current_x, current_y = y, x
        else:
            current_x, current_y = x, y

        if current_y < cutoff_height:
            continue

        if total_length > 0:
            dist = math.sqrt((current_x - x0) ** 2 + (current_y - y0) ** 2)
            t = min(1.0, max(0.0, dist / total_length))
        else:
            t = 0

        color = interpolate_color(color1, color2, t)

        for dx_offset in range(-thickness // 2, thickness // 2 + 1):
            for dy_offset in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = int(current_x + dx_offset), int(current_y + dy_offset)
                if 0 <= nx < w and 0 <= ny < h and ny >= cutoff_height:
                    if dx_offset * dx_offset + dy_offset * dy_offset <= (thickness // 2) ** 2:
                        image[ny, nx] = color

        error += dy_total
        if 2 * error >= dx_total:
            y += y_step
            error -= dx_total


# Создаем массив центров с деформацией
centers = []
for row in range(num_rows):
    row_centers = []
    for col in range(num_cols):
        base_x = offset_x + col * spacing
        base_y = offset_y + row * spacing

        if row == 0 or row == num_rows - 1 or col == 0 or col == num_cols - 1:
            center_x = base_x
            center_y = base_y
        else:
            center_x = base_x + random.randint(-polygon_size, polygon_size)
            center_y = base_y + random.randint(-polygon_size, polygon_size)

        row_centers.append((center_x, center_y))
    centers.append(row_centers)

# Создаем структуру данных для полигонов
convex_polygons = []
concave_polygons = []

for row in range(num_rows - 1):
    for col in range(num_cols - 1):
        top_left = centers[row][col]
        top_right = centers[row][col + 1]
        bottom_left = centers[row + 1][col]
        bottom_right = centers[row + 1][col + 1]

        # Цвета вершин полигона (ПРАВИЛЬНЫЙ ПОРЯДОК!)
        colors = [
            vertex_colors[(col, row)],  # top_left
            vertex_colors[(col + 1, row)],  # top_right
            vertex_colors[(col + 1, row + 1)],  # bottom_right
            vertex_colors[(col, row + 1)]  # bottom_left
        ]

        vertices = [top_left, top_right, bottom_right, bottom_left]

        # Сохраняем информацию о полигоне
        polygon = {
            'id': (col, row),
            'vertices': vertices,
            'colors': colors,
            'center': (
                (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) // 4,
                (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) // 4
            )
        }

        # Разделяем на выпуклые и невыпуклые
        if is_convex(vertices):
            convex_polygons.append(polygon)
        else:
            concave_polygons.append(polygon)

polygons = convex_polygons + concave_polygons

# Создаем изображение
image = np.full([h, w, 3], bg_color, dtype='uint8')

# Сбрасываем Z-буфер
z_buffer.fill(-float('inf'))

# Выбираем случайный полигон для закрашивания
filled_polygon = None
if polygons:
    # Выбираем только полигоны, которые полностью видны
    visible_polygons = [p for p in polygons if all(is_outside_parabola(v[0], v[1]) for v in p['vertices'])]

    if visible_polygons:
        filled_polygon = random.choice(visible_polygons)
        use_interpolation = FILL_MODE == "interpolate"

        # Определяем тип полигона и используем соответствующий алгоритм
        if filled_polygon in convex_polygons:
            fill_convex_polygon_bresenham(filled_polygon['vertices'], filled_polygon['colors'],
                                          image, use_interpolation)
            print(f"Закрашен выпуклый полигон {filled_polygon['id']} (метод Брезенхема)")
        else:
            fill_concave_polygon_zbuffer(filled_polygon['vertices'], filled_polygon['colors'],
                                         image, use_interpolation)
            print(f"Закрашен невыпуклый полигон {filled_polygon['id']} (Z-буфер)")

        print(f"Режим: {'интерполяция' if use_interpolation else 'один цвет'}")

# Рисуем линии полигонов с помощью алгоритма Брезенхема
for polygon in polygons:
    vertices = polygon['vertices']
    colors = polygon['colors']

    # Проверяем, что все вершины полигона находятся снаружи параболы
    all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in vertices)

    if all_vertices_outside:
        # Рисуем 4 стороны полигона
        for i in range(4):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 4]
            color1 = colors[i]
            color2 = colors[(i + 1) % 4]
            draw_line_bresenham(x1, y1, x2, y2, color1, color2, image, thickness=4)

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
plt.show()

# Выводим информацию о параметрах
print(f"Параметры сетки:")
print(f"Квадратов по X: {num_cols}")
print(f"Квадратов по Y: {num_rows}")
print(f"Размер полигона деформации: {polygon_size} пикселей")
print(f"Выпуклых полигонов: {len(convex_polygons)}")
print(f"Невыпуклых полигонов: {len(concave_polygons)}")
print(f"Режим закрашивания: {FILL_MODE}")