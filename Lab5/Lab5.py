import numpy as np
import matplotlib.pyplot as plt

from Lab4.Lab4 import GRID_MOD

# Настройки изображения
w = 600
h = 600
bg_color = (20, 20, 40)

# ПАРАМЕТРЫ ПАРАБОЛЫ
GRID_MOD = True
GRID_X = 25
GRID_Y = 25
MAX_Z = 200
Z_SCALE = 2.0
X_SCALE = 2.0
Y_SCALE = 2.0

# ПАРАМЕТРЫ ПАРАБОЛЫ
A_COEFF = 0.005
VERTEX_Y = 100
ROOF_Y = 0
PARABOLA_X_MAX = np.sqrt((VERTEX_Y - ROOF_Y) / A_COEFF)

# ПАРАМЕТРЫ УГЛА (в радианах)
ANGLE_X = np.radians(-6)
ANGLE_Y = np.radians(7)
ANGLE_Z = np.radians(-40)

# РЕЖИМ
WITH_DEPRESSION = False

# Z-буфер
z_buffer = None


def init_z_buffer():
    global z_buffer
    z_buffer = np.full((h, w), -np.inf, dtype=np.float32)


def create_rotation_matrix(angle_x, angle_y, angle_z):
    """Матрица поворота"""
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)

    rot_x = np.array([
        [1, 0, 0, 0],
        [0, cos_x, -sin_x, 0],
        [0, sin_x, cos_x, 0],
        [0, 0, 0, 1]
    ])

    rot_y = np.array([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, 0],
        [0, 0, 0, 1]
    ])

    rot_z = np.array([
        [cos_z, -sin_z, 0, 0],
        [sin_z, cos_z, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return rot_x @ rot_y @ rot_z


def is_inside_parabola(x_coord, y_coord):
    """Проверка, находится ли точка внутри параболы"""
    y_boundary = VERTEX_Y - A_COEFF * (x_coord / X_SCALE) ** 2
    y_boundary = max(ROOF_Y, y_boundary)
    return ROOF_Y <= y_coord / Y_SCALE <= y_boundary


def get_parabola_height(x, y, with_depression):
    """Высота параболы в точке"""
    xs = x / X_SCALE
    ys = y / Y_SCALE

    if not is_inside_parabola(x, y):
        return 0

    dist_norm = np.sqrt((xs / PARABOLA_X_MAX) ** 2 + (ys / VERTEX_Y) ** 2)
    dist_norm = min(1.0, dist_norm)
    z = max(0, MAX_Z * (1 - dist_norm))

    if with_depression:
        crater_radius = 30
        dist_xy = np.sqrt(xs ** 2 + ys ** 2)
        if dist_xy < crater_radius:
            z = 0

    return z


def create_paraboloid_grid():
    """Создает сетку параболоида"""
    vertices = []
    polygons = []
    vertex_grid = [[None for _ in range(GRID_X)] for _ in range(GRID_Y)]
    front_wall_bottom_points = []

    x_vals = np.linspace(-PARABOLA_X_MAX * X_SCALE, PARABOLA_X_MAX * X_SCALE, GRID_X)
    y_vals = np.linspace(0, VERTEX_Y * Y_SCALE, GRID_Y)

    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            z_val = get_parabola_height(x_val, y_val, WITH_DEPRESSION)
            vertices.append([x_val, y_val, z_val, 1])
            vertex_grid[i][j] = len(vertices) - 1

            # Добавляем точки для передней стенки
            if i == 0 and is_inside_parabola(x_val, y_val):
                vertices.append([x_val, 0, 0, 1])
                front_wall_bottom_points.append(len(vertices) - 1)

    # Основные полигоны параболоида
    for i in range(GRID_Y - 1):
        for j in range(GRID_X - 1):
            poly = [vertex_grid[i][j], vertex_grid[i][j + 1],
                    vertex_grid[i + 1][j + 1], vertex_grid[i + 1][j]]
            if None in poly:
                continue

            # Проверяем, что полигон внутри параболы
            mid_x = sum(vertices[idx][0] for idx in poly) / 4
            mid_y = sum(vertices[idx][1] for idx in poly) / 4
            if not is_inside_parabola(mid_x, mid_y):
                continue

            polygons.append(poly)

    # Передняя стенка
    for j in range(GRID_X - 1):
        if (vertex_grid[0][j] is not None and
                vertex_grid[0][j + 1] is not None and
                j < len(front_wall_bottom_points) - 1):
            poly = [vertex_grid[0][j], vertex_grid[0][j + 1],
                    front_wall_bottom_points[j + 1], front_wall_bottom_points[j]]
            polygons.append(poly)

    return np.array(vertices), polygons


def get_vertex_color(z_value):
    """Цвет вершины на основе высоты"""
    height_ratio = z_value / (MAX_Z * Z_SCALE)

    if height_ratio > 0.98:
        return 255
    else:
        return int(0 + height_ratio * 160)


def lerp(a, b, t):
    """Линейная интерполяция"""
    return a * (1 - t) + b * t


def draw_line_bresenham(image, x0, y0, x1, y1, color):
    """Рисование линии алгоритмом Брезенхема"""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1

    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    dx, dy = x1 - x0, abs(y1 - y0)
    error = -dx
    y_step = 1 if y0 < y1 else -1
    y = y0

    for x in range(x0, x1 + 1):
        plot_x, plot_y = (y, x) if steep else (x, y)

        if 0 <= plot_x < w and 0 <= plot_y < h:
            image[plot_y, plot_x] = color

        error += 2 * dy

        if error > 0:
            y += y_step
            error -= 2 * dx


def get_unique_edges(polygons):
    """Получение уникальных ребер"""
    edges = set()
    for poly in polygons:
        for i in range(len(poly)):
            idx1 = poly[i]
            idx2 = poly[(i + 1) % len(poly)]
            edges.add(tuple(sorted((idx1, idx2))))
    return edges


def draw_grid_without_depression(image, points, polygons, vertices):
    """Рисование сетки без линий в месте углубления"""
    edge_color = (255, 255, 255)
    edges = get_unique_edges(polygons)

    crater_radius = 30

    for idx1, idx2 in edges:
        p1 = points[idx1]
        p2 = points[idx2]
        v1 = vertices[idx1]
        v2 = vertices[idx2]

        # Проверяем, находятся ли обе вершины внутри углубления
        dist1 = np.sqrt((v1[0] / X_SCALE) ** 2 + (v1[1] / Y_SCALE) ** 2)
        dist2 = np.sqrt((v2[0] / X_SCALE) ** 2 + (v2[1] / Y_SCALE) ** 2)

        # Если обе вершины внутри углубления - не рисуем линию
        if WITH_DEPRESSION and dist1 < crater_radius and dist2 < crater_radius:
            continue

        draw_line_bresenham(
            image,
            p1[0] + w // 2,
            h // 2 - p1[1],
            p2[0] + w // 2,
            h // 2 - p2[1],
            edge_color
        )


def fill_polygon_z_buffer(image, poly_points, vertex_colors):
    """Закрашивание полигона с Z-буфером"""
    # Проекция на экран
    x_screen = poly_points[:, 0] + w // 2
    y_screen = h // 2 - poly_points[:, 1]

    poly_data = np.hstack((
        x_screen[:, None],
        y_screen[:, None],
        poly_points[:, 2][:, None],
        vertex_colors
    ))

    n = len(poly_data)
    min_y = int(max(0, np.min(y_screen)))
    max_y = int(min(h - 1, np.max(y_screen)))

    if min_y >= max_y:
        return

    for y in range(min_y, max_y + 1):
        intersections = []

        for i in range(n):
            p0, p1 = poly_data[i], poly_data[(i + 1) % n]
            y0, y1 = p0[1], p1[1]

            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0)
                x_int = lerp(p0[0], p1[0], t)
                z_int = lerp(p0[2], p1[2], t)
                color_int = lerp(p0[3:], p1[3:], t)
                intersections.append((x_int, z_int, color_int))

        intersections.sort(key=lambda item: item[0])

        for i in range(0, len(intersections), 2):
            if i + 1 >= len(intersections):
                break

            x_left, z_left, c_left = intersections[i]
            x_right, z_right, c_right = intersections[i + 1]
            span = x_right - x_left

            if span < 1e-6:
                continue

            for x in range(int(x_left), int(x_right) + 1):
                t_x = (x - x_left) / span
                z_current = lerp(z_left, z_right, t_x)
                color_current = lerp(c_left, c_right, t_x)
                buf_x, buf_y = int(x), int(y)

                if (0 <= buf_x < w and 0 <= buf_y < h and
                        z_current > z_buffer[buf_y, buf_x]):
                    z_buffer[buf_y, buf_x] = z_current
                    image[buf_y, buf_x] = np.clip(color_current, 0, 255).astype(np.uint8)


# ОСНОВНАЯ ПРОГРАММА
print("=== СОЗДАНИЕ ПАРАБОЛОИДА ===")
print(f"Режим: {'с углублением' if WITH_DEPRESSION else 'полный'}")
print(f"Углы: X={np.degrees(ANGLE_X):.1f}°, Y={np.degrees(ANGLE_Y):.1f}°, Z={np.degrees(ANGLE_Z):.1f}°")
print(f"Сетка: {GRID_X}x{GRID_Y}")

init_z_buffer()
image = np.full([h, w, 3], bg_color, dtype='uint8')

# Создаем и преобразуем вершины
vertices, polygons = create_paraboloid_grid()
vertices[:, 2] *= Z_SCALE

rotation_mat = create_rotation_matrix(ANGLE_X, ANGLE_Y, ANGLE_Z)
rotated_vertices = vertices @ rotation_mat.T

# Центрирование по вертикали
rotated_vertices[:, 1] -= 100

# Создаем цвета вершин
vertex_colors = []
for v in vertices:
    gray_val = get_vertex_color(v[2])
    vertex_colors.append([gray_val, gray_val, gray_val])

print(f"Создано вершин: {len(vertices)}, полигонов: {len(polygons)}")

# Закрашиваем полигоны
print("Закрашиваем полигоны...")
for poly in polygons:
    poly_points = rotated_vertices[poly][:, :3]
    poly_colors = np.array([vertex_colors[i] for i in poly], dtype=float)
    fill_polygon_z_buffer(image, poly_points, poly_colors)

# Рисуем сетку (без линий в углублении)
print("Рисуем сетку...")
if GRID_MOD:
   draw_grid_without_depression(image, rotated_vertices[:, :2], polygons, vertices)

print("Отображаем результат...")

# Отображаем
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title(f"Параболоид {'с углублением' if WITH_DEPRESSION else 'полный'}")
plt.tight_layout()
plt.show()

print("=== ГОТОВО ===")