import numpy as np
import matplotlib.pyplot as plt
import math
import os


def interpolate_along_line(start_pos, end_pos, start_d, end_d, start_col, end_col, start_tu=0.0, end_tu=1.0,
                           start_tv=0.0, end_tv=1.0):
    sx, sy = start_pos
    ex, ey = end_pos
    is_steep = abs(ey - sy) > abs(ex - sx)

    if is_steep:
        sx, sy = sy, sx
        ex, ey = ey, ex

    if sx > ex:
        sx, ex = ex, sx
        sy, ey = ey, sy
        start_d, end_d = end_d, start_d
        start_col, end_col = end_col, start_col
        start_tu, end_tu = end_tu, start_tu
        start_tv, end_tv = end_tv, start_tv

    delta_x = ex - sx
    if delta_x == 0:
        interpolated_pts = [(sx, y) for y in range(min(sy, ey), max(sy, ey) + 1)]
        num_pts = len(interpolated_pts)
        if num_pts == 0:
            return [], [], [], [], []
        interp_depths = np.linspace(start_d, end_d, num_pts)
        interp_colors = [start_col + (end_col - start_col) * t for t in np.linspace(0, 1, num_pts)]
        interp_tus = np.linspace(start_tu, end_tu, num_pts)
        interp_tvs = np.linspace(start_tv, end_tv, num_pts)
        if is_steep:
            interpolated_pts = [(y, x) for x, y in interpolated_pts]
        return interpolated_pts, interp_depths, interp_colors, interp_tus, interp_tvs

    delta_y_abs = abs(ey - sy)
    error = -delta_x
    y_step = 1 if sy < ey else -1
    current_y = sy

    current_depth = start_d
    depth_step = (end_d - start_d) / delta_x
    current_color = start_col.copy()
    color_step = (end_col - start_col) / delta_x
    current_tu = start_tu
    tu_step = (end_tu - start_tu) / delta_x
    current_tv = start_tv
    tv_step = (end_tv - start_tv) / delta_x

    interpolated_pts = []
    interp_depths = []
    interp_colors = []
    interp_tus = []
    interp_tvs = []

    for current_x in range(sx, ex + 1):
        if is_steep:
            actual_x, actual_y = current_y, current_x
        else:
            actual_x, actual_y = current_x, current_y
        interpolated_pts.append((actual_x, actual_y))
        interp_depths.append(current_depth)
        interp_colors.append(current_color.copy())
        interp_tus.append(current_tu)
        interp_tvs.append(current_tv)

        error += 2 * delta_y_abs
        if error > 0:
            current_y += y_step
            error -= 2 * delta_x

        current_depth += depth_step
        current_color += color_step
        current_tu += tu_step
        current_tv += tv_step

    return interpolated_pts, interp_depths, interp_colors, interp_tus, interp_tvs


# Настройки изображения
w = 600
h = 600
bg_color = (100, 100, 100)

# ПАРАМЕТРЫ ПАРАБОЛЫ
GRID_MOD = False
GRID_X = 15
GRID_Y = 15
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

# РЕЖИМЫ
# "normal" - обычная парабола, "hollow" - с углублением, "extrude" - с выдавливанием
MODE = "extrude"

# РЕЖИМ ТЕКСТУРЫ ДЛЯ ВЫДАВЛИВАНИЯ
TEXTURE_MODE = False  # True - с текстурой, False - без текстуры (градиент)

# ПАРАМЕТР ТЕКСТУРЫ - ИЗМЕНЯЕМЫЙ ПАРАМЕТР
TEXTURE_FILENAME = "images.jpg"  # Можно изменить на любой путь к изображению

# ПАРАМЕТРЫ ВЫДАВЛИВАНИЯ
EXTRUSION_UP_HEIGHT = 200  # Высота выдавливания вверх
EXTRUSION_DOWN_HEIGHT = -200  # Высота выдавливания вниз
EXTRUDE_UP_POLYGON = (13, 5)  # (row, col) - полигон для выдавливания вверх
EXTRUDE_DOWN_POLYGON = (0, 12)  # (row, col) - полигон для выдавливания вниз

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


def get_parabola_height(x, y):
    """Высота параболы в точке"""
    xs = x / X_SCALE
    ys = y / Y_SCALE

    if not is_inside_parabola(x, y):
        return 0

    dist_norm = np.sqrt((xs / PARABOLA_X_MAX) ** 2 + (ys / VERTEX_Y) ** 2)
    dist_norm = min(1.0, dist_norm)
    z = max(0, MAX_Z * (1 - dist_norm))

    return z


class TextureRenderer:
    def __init__(self, texture_filename=TEXTURE_FILENAME):
        self.texture_filename = texture_filename
        self.texture = None
        self.load_texture()

    def load_texture(self):
        """Загрузка текстуры из файла или генерация стандартной."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        local_path = os.path.join(dir_path, self.texture_filename)

        print(f"Попытка загрузки текстуры: {local_path}")

        if os.path.exists(local_path):
            print(f"Загружена текстура из {local_path}")
            img = plt.imread(local_path)
            if img.dtype == np.float64 or img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            self.texture = img
            print(f"Размер текстуры: {self.texture.shape}")
        else:
            print(f"Файл {self.texture_filename} не найден. Используется стандартная текстура.")
            self._generate_default_texture()

    def _generate_default_texture(self):
        """Генерация стандартной текстуры"""
        H = 256
        W = 256
        img = np.full((H, W, 3), 200, dtype=np.uint8)
        s = H // 4
        # Создаем разноцветные квадраты
        img[:s, s:2 * s, :2] = 0
        img[:s, 3 * s:, 2] = 0
        img[s:2 * s, :s, 0] = 0
        img[s:2 * s, 2 * s:3 * s, 1] = 0
        img[2 * s:3 * s, s:2 * s, 2] = 0
        img[2 * s:3 * s, 3 * s:, 0] = 0
        img[2 * s:3 * s, 3 * s:, 2] = 0
        img[3 * s:, :s, :2] = 100
        img[3 * s:, 2 * s:3 * s, 2] = 100
        self.texture = img

    def sample_texture_nearest(self, u: float, v: float) -> np.ndarray:
        """Сэмплирование текстуры nearest."""
        if self.texture is None:
            return np.array([128, 128, 128], dtype=np.uint8)
        H, W = self.texture.shape[:2]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        iu = int(u * (W - 1))
        iv = int((1 - v) * (H - 1))
        return self.texture[iv, iu]


def create_paraboloid_grid():
    """Создает сетку параболоида с возможностью выдавливания"""
    vertices = []
    polygons = []
    vertex_grid = [[None for _ in range(GRID_X)] for _ in range(GRID_Y)]
    front_wall_bottom_points = []

    x_vals = np.linspace(-PARABOLA_X_MAX * X_SCALE, PARABOLA_X_MAX * X_SCALE, GRID_X)
    y_vals = np.linspace(0, VERTEX_Y * Y_SCALE, GRID_Y)

    # Создаем базовые вершины
    for i, y_val in enumerate(y_vals):
        for j, x_val in enumerate(x_vals):
            if MODE == "extrude":
                z_val = 80  # постоянная высота для плоской поверхности
            else:
                z_val = get_parabola_height(x_val, y_val)

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

    # РЕЖИМ ВЫДАВЛИВАНИЯ
    if MODE == "extrude":
        print("Режим выдавливания: создаем дополнительные вершины...")

        up_poly_i, up_poly_j = EXTRUDE_UP_POLYGON
        down_poly_i, down_poly_j = EXTRUDE_DOWN_POLYGON

        if (up_poly_i < 0 or up_poly_i >= GRID_Y - 1 or up_poly_j < 0 or up_poly_j >= GRID_X - 1 or
                down_poly_i < 0 or down_poly_i >= GRID_Y - 1 or down_poly_j < 0 or down_poly_j >= GRID_X - 1):
            print("Ошибка: указаны неверные координаты полигонов для выдавливания")
            return np.array(vertices), polygons

        # Создаем вершины для выдавливания вверх
        top_up_indices = []
        base_up_indices = []
        for di in [0, 1]:
            for dj in [0, 1]:
                base_idx = vertex_grid[up_poly_i + di][up_poly_j + dj]
                if base_idx is None:
                    continue
                base_up_indices.append(base_idx)
                x, y, z, _ = vertices[base_idx]
                vertices.append([x, y, z + EXTRUSION_UP_HEIGHT, 1])
                top_up_indices.append(len(vertices) - 1)

        # Создаем вершины для выдавливания вниз
        top_down_indices = []
        base_down_indices = []
        for di in [0, 1]:
            for dj in [0, 1]:
                base_idx = vertex_grid[down_poly_i + di][down_poly_j + dj]
                if base_idx is None:
                    continue
                base_down_indices.append(base_idx)
                x, y, z, _ = vertices[base_idx]
                vertices.append([x, y, z + EXTRUSION_DOWN_HEIGHT, 1])
                top_down_indices.append(len(vertices) - 1)

        # Удаляем оригинальные полигоны
        polygons_to_remove = []
        for idx, poly in enumerate(polygons):
            is_up_extrusion = all(vertex_idx in base_up_indices for vertex_idx in poly)
            is_down_extrusion = all(vertex_idx in base_down_indices for vertex_idx in poly)

            if is_up_extrusion or is_down_extrusion:
                polygons_to_remove.append(idx)

        polygons_to_remove = sorted(set(polygons_to_remove), reverse=True)
        for idx in polygons_to_remove:
            polygons.pop(idx)

        # Добавляем новые полигоны для выдавливания

        # Верхняя крышка
        if len(top_up_indices) == 4:
            polygons.append([top_up_indices[0], top_up_indices[1], top_up_indices[3], top_up_indices[2]])

        # Нижняя крышка
        if len(top_down_indices) == 4:
            polygons.append([top_down_indices[0], top_down_indices[2], top_down_indices[3], top_down_indices[1]])

        # Боковые грани для выдавливания вверх
        if len(base_up_indices) == 4 and len(top_up_indices) == 4:
            polygons.append([base_up_indices[0], base_up_indices[1], top_up_indices[1], top_up_indices[0]])
            polygons.append([base_up_indices[1], base_up_indices[2], top_up_indices[3], top_up_indices[1]])
            polygons.append([base_up_indices[2], base_up_indices[3], top_up_indices[2], top_up_indices[3]])
            polygons.append([base_up_indices[3], base_up_indices[0], top_up_indices[0], top_up_indices[2]])

        # Боковые грани для выдавливания вниз
        if len(base_down_indices) == 4 and len(top_down_indices) == 4:
            polygons.append([base_down_indices[0], top_down_indices[0], top_down_indices[1], base_down_indices[1]])
            polygons.append([base_down_indices[1], top_down_indices[1], top_down_indices[3], base_down_indices[2]])
            polygons.append([base_down_indices[2], top_down_indices[3], top_down_indices[2], base_down_indices[3]])
            polygons.append([base_down_indices[3], top_down_indices[2], top_down_indices[0], base_down_indices[0]])

    elif MODE == "hollow":
        crater_radius = 30
        for v in vertices:
            x, y, z, _ = v
            xs = x / X_SCALE
            ys = y / Y_SCALE
            dist_xy = np.sqrt(xs ** 2 + ys ** 2)
            if dist_xy < crater_radius:
                v[2] = 0

    return np.array(vertices), polygons


def get_vertex_color(z_value):
    """Цвет вершины на основе высоты"""
    height_ratio = z_value / (MAX_Z * Z_SCALE)

    if height_ratio > 1.3:
        return 255
    elif height_ratio > 0.98:
        return 200
    elif height_ratio < 0.2:
        return 50
    else:
        return int(80 + height_ratio * 120)


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


def draw_grid(image, points, polygons):
    """Рисование сетки, исключая ребра в местах выдавливаний"""
    edge_color = (255, 255, 255)
    edges = get_unique_edges(polygons)

    # Находим индексы вершин, которые принадлежат выдавливаниям
    extrusion_vertex_indices = set()

    if MODE == "extrude":
        # Получаем базовые индексы для выдавливаний
        up_poly_i, up_poly_j = EXTRUDE_UP_POLYGON
        down_poly_i, down_poly_j = EXTRUDE_DOWN_POLYGON

        # Добавляем базовые вершины выдавливаний
        for di in [0, 1]:
            for dj in [0, 1]:
                # Вычисляем индексы базовых вершин для выдавливания вверх
                base_i = up_poly_i + di
                base_j = up_poly_j + dj
                if 0 <= base_i < GRID_Y and 0 <= base_j < GRID_X:
                    base_idx = base_i * GRID_X + base_j
                    extrusion_vertex_indices.add(base_idx)

                # Вычисляем индексы базовых вершин для выдавливания вниз
                base_i = down_poly_i + di
                base_j = down_poly_j + dj
                if 0 <= base_i < GRID_Y and 0 <= base_j < GRID_X:
                    base_idx = base_i * GRID_X + base_j
                    extrusion_vertex_indices.add(base_idx)

    for idx1, idx2 in edges:
        # Проверяем, принадлежат ли обе вершины ребра к выдавливанию
        if idx1 in extrusion_vertex_indices and idx2 in extrusion_vertex_indices:
            continue  # Пропускаем ребро, если обе вершины принадлежат выдавливанию

        p1 = points[idx1]
        p2 = points[idx2]

        draw_line_bresenham(
            image,
            p1[0] + w // 2,
            h // 2 - p1[1],
            p2[0] + w // 2,
            h // 2 - p2[1],
            edge_color
        )


def apply_rotation_to_vertices(vertices, rotation_mat):
    """Применяет поворот только к координатам x, y, z, сохраняя остальные данные"""
    rotated_vertices = vertices.copy()

    for i in range(len(vertices)):
        # Берем только координаты x, y, z, w
        xyz = vertices[i, :4]
        # Применяем поворот
        rotated_xyz = rotation_mat @ xyz
        # Нормализуем homogeneous coordinates
        rotated_xyz = rotated_xyz / rotated_xyz[3]
        # Записываем обратно только x, y, z (w остается 1)
        rotated_vertices[i, :3] = rotated_xyz[:3]

    return rotated_vertices


def generate_texture_coords_for_polygon(polygon_vertices):
    """Генерирует текстурные координаты для полигона - одна текстура на весь полигон"""
    # Получаем bounding box полигона в мировых координатах
    min_x = np.min(polygon_vertices[:, 0])
    max_x = np.max(polygon_vertices[:, 0])
    min_y = np.min(polygon_vertices[:, 1])
    max_y = np.max(polygon_vertices[:, 1])

    texture_coords = []
    for vertex in polygon_vertices:
        # Нормализуем координаты вершины относительно bounding box полигона
        if max_x - min_x > 0:
            u = (vertex[0] - min_x) / (max_x - min_x)
        else:
            u = 0.5

        if max_y - min_y > 0:
            v = (vertex[1] - min_y) / (max_y - min_y)
        else:
            v = 0.5

        texture_coords.append([u, v])

    return np.array(texture_coords)


def is_white_color(color, threshold=250):
    """Проверяет, является ли цвет белым (или близким к белому)"""
    return (color[0] >= threshold and
            color[1] >= threshold and
            color[2] >= threshold)


def fill_polygon_z_buffer(image, poly_points, vertex_colors, texture_renderer=None):
    """Закрашивание полигона с Z-буфером"""
    # Проекция на экран
    x_screen = poly_points[:, 0] + w // 2
    y_screen = h // 2 - poly_points[:, 1]

    # Генерируем текстурные координаты для всего полигона
    if TEXTURE_MODE and texture_renderer is not None:
        texture_coords = generate_texture_coords_for_polygon(poly_points)
    else:
        texture_coords = None

    # Создаем полигонные данные
    n_vertices = len(poly_points)
    poly_data = []

    for i in range(n_vertices):
        point_data = [x_screen[i], y_screen[i], poly_points[i, 2]]
        point_data.extend(vertex_colors[i])
        if texture_coords is not None:
            point_data.extend(texture_coords[i])
        poly_data.append(point_data)

    poly_data = np.array(poly_data)

    n = len(poly_data)
    min_y = int(max(0, np.min(y_screen)))
    max_y = int(min(h - 1, np.max(y_screen)))

    if min_y >= max_y:
        return

    for y in range(min_y, max_y + 1):
        intersections = []

        for i in range(n):
            p0 = poly_data[i]
            p1 = poly_data[(i + 1) % n]
            y0, y1 = p0[1], p1[1]

            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0) if (y1 - y0) != 0 else 0
                x_int = lerp(p0[0], p1[0], t)
                z_int = lerp(p0[2], p1[2], t)

                if TEXTURE_MODE and texture_renderer is not None and texture_coords is not None:
                    # Интерполируем текстурные координаты
                    tu_int = lerp(p0[6], p1[6], t)  # tu на позиции 6
                    tv_int = lerp(p0[7], p1[7], t)  # tv на позиции 7
                    intersections.append((x_int, z_int, tu_int, tv_int))
                else:
                    # Интерполируем цвет
                    color_int = [
                        lerp(p0[3], p1[3], t),  # R
                        lerp(p0[4], p1[4], t),  # G
                        lerp(p0[5], p1[5], t)  # B
                    ]
                    intersections.append((x_int, z_int, color_int))

        intersections.sort(key=lambda item: item[0])

        for i in range(0, len(intersections), 2):
            if i + 1 >= len(intersections):
                break

            if TEXTURE_MODE and texture_renderer is not None:
                x_left, z_left, tu_left, tv_left = intersections[i]
                x_right, z_right, tu_right, tv_right = intersections[i + 1]
            else:
                x_left, z_left, c_left = intersections[i]
                x_right, z_right, c_right = intersections[i + 1]

            span = x_right - x_left

            if span < 1e-6:
                continue

            for x in range(int(x_left), int(x_right) + 1):
                t_x = (x - x_left) / span if span != 0 else 0
                z_current = lerp(z_left, z_right, t_x)
                buf_x, buf_y = int(x), int(y)

                if (0 <= buf_x < w and 0 <= buf_y < h and
                        z_current > z_buffer[buf_y, buf_x]):
                    z_buffer[buf_y, buf_x] = z_current

                    if TEXTURE_MODE and texture_renderer is not None:
                        tu_current = lerp(tu_left, tu_right, t_x)
                        tv_current = lerp(tv_left, tv_right, t_x)
                        tex_color = texture_renderer.sample_texture_nearest(tu_current, tv_current)

                        # ← ДОБАВЛЕНА ПРОВЕРКА НА БЕЛЫЙ ЦВЕТ!
                        if not is_white_color(tex_color):
                            image[buf_y, buf_x] = tex_color
                        # Если цвет белый - не рисуем (оставляем фон)
                    else:
                        color_current = [
                            lerp(c_left[0], c_right[0], t_x),
                            lerp(c_left[1], c_right[1], t_x),
                            lerp(c_left[2], c_right[2], t_x)
                        ]
                        image[buf_y, buf_x] = np.clip(color_current, 0, 255).astype(np.uint8)


# ОСНОВНАЯ ПРОГРАММА
print("=== СОЗДАНИЕ ПАРАБОЛОИДА С ВЫДАВЛИВАНИЕМ ===")
mode_titles = {
    "normal": "Обычная парабола",
    "hollow": "Парабола с углублением",
    "extrude": f"ПЛОСКАЯ парабола с выдавливанием"
}
print(f"Режим: {mode_titles[MODE]}")
print(f"Режим текстуры: {'ВКЛЮЧЕН' if TEXTURE_MODE else 'ВЫКЛЮЧЕН'}")
print(f"Текстура: {TEXTURE_FILENAME}")
print(f"Углы: X={np.degrees(ANGLE_X):.1f}°, Y={np.degrees(ANGLE_Y):.1f}°, Z={np.degrees(ANGLE_Z):.1f}°")
print(f"Сетка: {GRID_X}x{GRID_Y}")

init_z_buffer()
image = np.full([h, w, 3], bg_color, dtype='uint8')

# Инициализация текстурного рендерера если нужно
texture_renderer = None
if TEXTURE_MODE:
    texture_renderer = TextureRenderer(TEXTURE_FILENAME)

# Создаем вершины
vertices, polygons = create_paraboloid_grid()

# Применяем масштаб по Z и вращение
vertices[:, 2] *= Z_SCALE

rotation_mat = create_rotation_matrix(ANGLE_X, ANGLE_Y, ANGLE_Z)
rotated_vertices = apply_rotation_to_vertices(vertices, rotation_mat)

# Центрирование по вертикали
rotated_vertices[:, 1] -= 100

# Создаем цвета вершин
vertex_colors = []
for v in vertices:
    if TEXTURE_MODE:
        # Для режима текстуры используем белый цвет (будет заменен текстурой)
        gray_val = 255
        vertex_colors.append([gray_val, gray_val, gray_val])
    else:
        # Для режима без текстуры используем градиент по высоте
        gray_val = get_vertex_color(v[2])
        vertex_colors.append([gray_val, gray_val, gray_val])

vertex_colors = np.array(vertex_colors, dtype=float)

print(f"Создано вершин: {len(vertices)}, полигонов: {len(polygons)}")

# Закрашиваем полигоны
print("Закрашиваем полигоны...")
for poly in polygons:
    poly_points = rotated_vertices[poly][:, :3]
    poly_colors = vertex_colors[poly]

    fill_polygon_z_buffer(image, poly_points, poly_colors, texture_renderer)

# Рисуем сетку
print("Рисуем сетку...")
if GRID_MOD is True:
   draw_grid(image, rotated_vertices[:,:2], polygons)

print("Отображаем результат...")

# Отображаем
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title(
    f"{mode_titles[MODE]}\nТекстура: {'ВКЛ' if TEXTURE_MODE else 'ВЫКЛ'} | Углы: X={np.degrees(ANGLE_X):.1f}°, Y={np.degrees(ANGLE_Y):.1f}°, Z={np.degrees(ANGLE_Z):.1f}°")
plt.tight_layout()
plt.show()

print("=== ГОТОВО ===")