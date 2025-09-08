import numpy as np
import matplotlib.pyplot as plt


class DrawCanvas:
    def __init__(self):
        self.HEIGHT = self.WIDTH = 600
        self.colour = (200, 200, 200)
        self.border = 5
        self.borderColour = (255, 0, 0)
        self.canvas = np.full([self.HEIGHT, self.WIDTH, 3], self.colour, dtype="uint")


    def drawRectangle(self, border, borderColour, x, y, w, h):
        self.canvas[y:y + h, x:x + border] = borderColour  # отрисовка левого бока
        self.canvas[y:y + h, x + w - border: x + w] = borderColour  # отрисовка правого бока
        self.canvas[y:y + border, x:x + w] = borderColour  # отрисовка верха
        self.canvas[y + h - border:y + h, x:x + w] = borderColour  # отрисовка низа

    def drawBorder(self):
        self.drawRectangle(self.border, self.borderColour, 0, 0, self.WIDTH, self.HEIGHT)

    def drawRectanglesInCanvas(self):
        a = b = 300
        radius = 50
        coord = (self.WIDTH - a) // 2 - (radius // 2)
        for x in [coord, coord + a]:
            for y in [coord, coord + b]:
                self.drawRectangle(self.border, self.borderColour, x, y, radius, radius)

    def drawPoint(self, x, y, colour):
        self.drawRectangle(5, colour, x, y, 10, 10)

    def drawRandomPoints(self):
        #  try
        pass

if __name__ == "__main__":
    dc = DrawCanvas()
    dc.drawBorder()
    dc.drawRectanglesInCanvas()
    plt.imshow(dc.canvas)
    plt.axis("off")
    plt.show()