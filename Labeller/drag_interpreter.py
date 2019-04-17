from .geometry import Point, Rectangle



class DragInterpreterBase:
    def __init__(self):
        self.__on_dragging = False


    @property
    def on_dragging(self):
        return self.__on_dragging


    def start_dragging(self, p: Point):
        if p != None:
            self.__on_dragging = True


    def finish_dragging(self, p: Point):
        self.__on_dragging = False



class DragInterpreter(DragInterpreterBase):
    def __init__(self):
        super(DragInterpreter, self).__init__()
        self.rect = Rectangle()
        self.p1 = self.rect.tl_corner
        self.p2 = self.rect.br_corner


    def infer_rect(self):
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
        if p != None:
            self.p1 = p
            self.p2 = p
            self.rect = self.infer_rect()


    def update(self, p: Point):
        if p != None:
            self.p2 = p
            self.rect = self.infer_rect()


    def finish_dragging(self, p: Point):
        super(DragInterpreter, self).finish_dragging(p)
        self.update(p)


