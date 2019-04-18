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
        super(DragInterpreter, self).__init__()
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
        super(DragInterpreter, self).start_dragging(p)
        if p is not None:
            self.p1 = p
            self.p2 = p

    def update(self, p: Point):
        if p is not None:
            self.p2 = p

    def finish_dragging(self, p: Point):
        super(DragInterpreter, self).finish_dragging(p)
        self.update(p)
