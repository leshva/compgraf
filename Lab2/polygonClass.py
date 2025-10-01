from settings import *


class Polygon:
    def __init__(self, row, col):
        self.top_left = CENTERS[row][col]
        self.top_right = CENTERS[row][col + 1]
        self.bottom_left = CENTERS[row + 1][col]
        self.bottom_right = CENTERS[row + 1][col + 1]