from random import randint

# Настройки изображения
WIDTH = 600
HEIGHT = 600
BG_COLOR = (150, 150, 150)  # серый фон

# ИЗМЕНЯЕМЫЕ ПАРАМЕТРЫ
NUM_COLS = 10  # число квадратов по X
NUM_ROWS = 10 # число квадратов по Y
POLYGON_SIZE = 20 # размер полигона (области деформации)

# Параметры закрашивания (можно менять)
FILL_MODE = "interpolate"  # "solid" - одним цветом, "interpolate" - интерполяция цветов
FILL_ALL_RECTANGLES = True  # True - закрасить все прямоугольники, False - только один случайный

# Автоматический расчет spacing на основе числа квадратов
SPACING_X = WIDTH // (NUM_COLS - 1) if NUM_COLS > 1 else WIDTH
SPACING_Y = HEIGHT // (NUM_ROWS - 1) if NUM_ROWS > 1 else HEIGHT
SPACING = min(SPACING_X, SPACING_Y)

# Параметры квадратов
SQUARE_SIZE = 25

OFFSET_X = (WIDTH - (NUM_COLS - 1) * SPACING) // 2
OFFSET_Y = (HEIGHT - (NUM_ROWS - 1) * SPACING) // 2

# Параметры перевернутой параболы (вершина в центре)
A = -0.007  # отрицательный коэффициент для перевернутой параболы
VERTEX_X = WIDTH // 2
VERTEX_Y = HEIGHT // 2 + 300
CUTOFF_HEIGHT = 10  # Высота, ниже которой всё исчезает

# Генерируем случайные цвета для вершин
VERTEX_COLORS = {}
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        r = randint(50, 255)
        g = randint(50, 255)
        b = randint(50, 255)
        VERTEX_COLORS[(col, row)] = (r, g, b)

# Создаем массив центров с деформацией
CENTERS = []
for row in range(NUM_ROWS):
    row_centers = []
    for col in range(NUM_COLS):
        base_x = OFFSET_X + col * SPACING
        base_y = OFFSET_Y + row * SPACING

        if row == 0 or row == NUM_ROWS - 1 or col == 0 or col == NUM_COLS - 1:
            center_x = base_x
            center_y = base_y
        else:
            center_x = base_x + randint(-POLYGON_SIZE, POLYGON_SIZE)
            center_y = base_y + randint(-POLYGON_SIZE, POLYGON_SIZE)

        row_centers.append((center_x, center_y))
    CENTERS.append(row_centers)


# Создаем структуру данных для прямоугольников
POLYGONS = []

for row in range(NUM_ROWS - 1):
    for col in range(NUM_COLS - 1):
        top_left = CENTERS[row][col]
        top_right = CENTERS[row][col + 1]
        bottom_left = CENTERS[row + 1][col]
        bottom_right = CENTERS[row + 1][col + 1]

        # Цвета вершин прямоугольника
        colors = [
            VERTEX_COLORS[(col, row)],  # top_left
            VERTEX_COLORS[(col + 1, row)],  # top_right
            VERTEX_COLORS[(col + 1, row + 1)],  # bottom_right
            VERTEX_COLORS[(col, row + 1)]  # bottom_left
        ]

        vertices = [top_left, top_right, bottom_right, bottom_left]

        # Сохраняем информацию о прямоугольнике
        polygon = {
            'id': (col, row),
            'vertices': vertices,
            'colors': colors,
            'center': (
                (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) // 4,
                (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) // 4
            )
        }

        POLYGONS.append(polygon)