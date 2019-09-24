import sys
import numpy as np
import cv2


WIN_NAME = 'WINDOW'


class Renderer:
    WHITE = (1, 1, 1)
    RED = (0, 0, 1)

    def __init__(self, w, h):
        self.canvas = np.ones((w, h, 3))

        self.width = w
        self.height = h

        self.origin_x = self.width / 2
        self.origin_y = self.height / 2
        self.f = 0

    def __call__(self, loop):
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 800, 800)
        cv2.moveWindow(WIN_NAME, 100, 100)

        while True:
            self.f += 1
            cv2.imshow(WIN_NAME, self.canvas)
            self.clear()

            if cv2.waitKey(33) == -1:
                try:
                    loop(self)
                except Exception as e:
                    self.destroy()
                    raise e
            else:
                self.destroy()
                break

    def destroy(self):
        cv2.destroyWindow(WIN_NAME)

    def clear(self):
        self.canvas = np.zeros((self.width, self.height, 3))

    def rect(self, x, y, w, h, rgb=(1, 1, 1), thickness=-1):
        x, y = self._origin_translate(x, y)
        cv2.rectangle(
            self.canvas,
            (int(x), int(y)), (int(x + w), int(y + h)), rgb, thickness=thickness
        )

    def arc(self, x, y, rad, rgb=(1, 1, 1), thickness=-1):
        x, y = self._origin_translate(x, y)
        cv2.circle(
            self.canvas, (int(x), int(y)),
            int(rad), rgb, thickness=thickness
        )

    def _origin_translate(self, x, y):
        return self.origin_x + x, self.origin_y - y
