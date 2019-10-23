import numpy as np
from .geometry import Point, Rectangle


class DragInterpreterBase:
    def __init__(self):
        self.__on_dragging = False

    @property
    def on_dragging(self):
        return self.__on_dragging

    def start_dragging(self, p: Point):
        if p is not None:
            self.__on_dragging = True

    def finish_dragging(self, p: Point):
        self.__on_dragging = False


class DragInterpreter(DragInterpreterBase):
    def __init__(self):
        super().__init__()
        self.p1 = Point(0, 0)
        self.p2 = Point(-1, -1)

    @property
    def rect(self):
        tl = Point(
            min(self.p1.x, self.p2.x),
            min(self.p1.y, self.p2.y)
        )
        br = Point(
            max(self.p1.x, self.p2.x),
            max(self.p1.y, self.p2.y)
        )
        return Rectangle(tl, br)

    def start_dragging(self, p: Point):
        super().start_dragging(p)
        if p is not None:
            self.p1 = p
            self.p2 = p

    def update(self, p: Point):
        if p is not None:
            self.p2 = p

    def finish_dragging(self, p: Point):
        super().finish_dragging(p)
        self.update(p)


class PolarDragInterpreter(DragInterpreter):
    def __init__(self):
        super().__init__()
        self.anticlockwise = True

    def start_dragging(self, p: Point, anticlockwise=None):
        super().start_dragging(p)
        if anticlockwise is not None:
            self.anticlockwise = anticlockwise

    def update(self, p: Point, anticlockwise=None):
        super().update(p)
        if anticlockwise is not None:
            self.anticlockwise = anticlockwise

    def finish_dragging(self, p: Point, anticlockwise=None):
        super().finish_dragging(p)
        if anticlockwise is not None:
            self.anticlockwise = anticlockwise

    @property
    def rect(self):
        if self.anticlockwise and self.p1.x < self.p2.x:
            theta_begin, theta_end = self.p1.x, self.p2.x
        elif self.anticlockwise and self.p1.x > self.p2.x:
            theta_begin, theta_end = self.p1.x - 2 * np.pi, self.p2.x
        elif not self.anticlockwise and self.p1.x < self.p2.x:
            theta_begin, theta_end = self.p2.x - 2 * np.pi, self.p1.x
        else:
            theta_begin, theta_end = self.p2.x, self.p1.x

        tl = Point(
            theta_begin,
            min(256, min(self.p1.y, self.p2.y)),
            dtype=float
        )
        br = Point(
            theta_end,
            min(256, max(self.p1.y, self.p2.y)),
            dtype=float
        )
        return Rectangle(tl, br, dtype=float)
