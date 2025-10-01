from settings import *


class CenterWithDeformation:
    def __init__(self, row, col):
        base_x, base_y = OFFSET_X + col * SPACING, OFFSET_Y + row * SPACING

        if row == 0 or row == NUM_ROWS - 1 or col == 0 or col == NUM_COLS - 1:
            center_x = base_x
            center_y = base_y
        else:
            center_x = base_x + randint(-POLYGON_SIZE, POLYGON_SIZE)
            center_y = base_y + randint(-POLYGON_SIZE, POLYGON_SIZE)

        self.center_x = center_x
        self.center_y = center_y