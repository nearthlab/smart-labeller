import copy
import enum

import cv2
import numpy as np
from matplotlib import patches as patches

from Labeller.drag_interpreter import DragInterpreterBase
from Labeller.geometry import Point


class BrushType(enum.Enum):
    # val = pixel value for grabcut mask
    # color = BGR pixel value for visualization
    BG = {'val': 0, 'color': (0, 0, 0)}
    FG = {'val': 1, 'color': (0, 255, 0)}
    PR_BG = {'val': 2, 'color': (51, 51, 153)}
    PR_FG = {'val': 3, 'color': (51, 255, 204)}

    @classmethod
    def val2color(cls, val):
        return {
            brush_type.value['val']: np.array(brush_type.value['color'])
            for brush_type in cls.__members__.values()
        }[val]

    @classmethod
    def val2name(cls, val):
        return {
            brush_type.value['val']: brush_type.name
            for brush_type in cls.__members__.values()
        }[val]

    def __add__(self, other: int):
        val = (self.value['val'] + other) % len(BrushType)
        return {
            brush_type.value['val']: brush_type
            for brush_type in BrushType.__members__.values()
        }[val]

    def __sub__(self, other: int):
        return self.__add__(-other)


class BrushInterpreter(DragInterpreterBase):
    def __init__(self):
        super().__init__()
        self.radius = 30
        self.brush = BrushType.BG
        self.brush_trace = []

    def get_trace(self, p):
        trace = BrushTouch(p, self.radius, True, self.brush)
        self.brush_trace.append(trace)
        return trace

    def history(self):
        return copy.deepcopy(self.brush_trace)

    def clear(self):
        self.brush_trace.clear()


class BrushTouch:
    def __init__(self, center: Point, radius: int, solid: bool, brush: BrushType):
        self.center = center
        self.radius = radius
        self.solid = solid
        self.brush = brush

    def patch(self, alpha=1.0):
        color = [val / 255 for val in self.brush.value['color']]
        color.append(alpha)
        color = tuple(color)
        return patches.Circle(
            tuple(self.center), self.radius,
            edgecolor=color if alpha == 1.0 else 'w',
            facecolor=color if self.solid else 'none',
            linewidth=2
        )


def apply_brush_touch(img: np.ndarray, brush_touch: BrushTouch):
    key = 'val' if img.ndim == 2 else 'color'
    return cv2.circle(img, tuple(brush_touch.center), brush_touch.radius, brush_touch.brush.value[key], -1 if brush_touch.solid else 1)
