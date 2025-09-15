import numpy as np
import matplotlib.pyplot as plt

from random import choice
from functions import *


def main():
    # Создаем изображение
    image = np.full([HEIGHT, WIDTH, 3], BG_COLOR, dtype='uint8')

    use_interpolation = FILL_MODE == "interpolate"

    # Сначала закрашиваем все прямоугольники
    filled_count = 0

    if FILL_ALL_RECTANGLES:
        # Закрашиваем ВСЕ прямоугольники, которые находятся снаружи параболы
        for polygon in POLYGONS:
            # Проверяем, что все вершины прямоугольника находятся снаружи параболы
            all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in polygon['vertices'])

            if all_vertices_outside:
                fill_rectangle_bresenham(polygon['vertices'], polygon['colors'], image, use_interpolation)
                filled_count += 1
    else:
        # Закрашиваем только ОДИН СЛУЧАЙНЫЙ прямоугольник
        valid_polygons = []
        for polygon in POLYGONS:
            # Проверяем, что все вершины прямоугольника находятся снаружи параболы
            all_vertices_outside = all(is_outside_parabola(v[0], v[1]) for v in polygon['vertices'])
            if all_vertices_outside:
                valid_polygons.append(polygon)

        if valid_polygons:
            # Выбираем случайный прямоугольник для закрашивания
            random_polygon = choice(valid_polygons)
            fill_rectangle_bresenham(random_polygon['vertices'], random_polygon['colors'], image, use_interpolation)
            filled_count = 1
            print(f"Закрашен случайный прямоугольник с ID: {random_polygon['id']}")
        else:
            print("Нет подходящих прямоугольников для закрашивания")

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