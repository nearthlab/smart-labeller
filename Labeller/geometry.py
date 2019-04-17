import cv2
import numpy as np
from matplotlib import patches as patches
from skimage import measure
import shapely.geometry as geom



class Point:
    def __init__(self, *args, dtype=int):
        if dtype not in [int, float]:
            raise ValueError('dtype must be either int or float but {} is given'.format(dtype))
        if len(args) == 0:
            data = np.zeros((2))
        elif len(args) == 1:
            data = np.array(args[0])
            if data.ndim != 1 or data.shape[0] != 2:
                raise ValueError('Invalid argument: {}'.format(args[0]))
        elif len(args) == 2:
            data = np.array(args)
        else:
            raise ValueError('Invalid arguments: {}'.format(args))
        self.__data = data.astype(np.float) if dtype == float \
                else data.round().astype(np.int)

    @property
    def x(self):
        return self.__data[0]

    @property
    def y(self):
        return self.__data[1]

    @x.setter
    def x(self, x_):
        self.__data[0] = x_

    @y.setter
    def y(self, y_):
        self.__data[1] = y_

    @property
    def dtype(self):
        return self.__data.dtype


    def astype(self, dtype):
        if dtype not in [int, float]:
            raise ValueError('dtype must be either int or float but {} is given'.format(dtype))
        else:
            return Point(self.__data, dtype=dtype)


    def length_squared(self):
        return np.sum(np.square(self.__data)).item()


    def length(self):
        return np.sqrt(self.length_squared()).item()


    def normalize(self):
        return Point(self.__data / self.length(), dtype=float)


    def __str__(self):
        return 'Point({}, {})'.format(self.x, self.y)


    def __repr__(self):
        return 'Point({}, {})'.format(self.x, self.y)


    def __eq__(self, other):
        if isinstance(other, Point):
            return np.array_equal(self.__data, other.__data)
        else: return False


    def __add__(self, other):
        return Point(self.__data + other.__data, dtype=self.dtype)


    def __sub__(self, other):
        return Point(self.__data - other.__data, dtype=self.dtype)


    def __mul__(self, other):
        if type(other) not in [int, float]:
            raise TypeError('Invalid scalar argument: {} (type={})'.format(other, type(other)))
        return Point(other * self.__data, dtype=np.float)


    def __rmul__(self, other):
        if type(other) not in [int, float]:
            raise TypeError('Invalid scalar argument: {} (type={})'.format(other, type(other)))
        return Point(other * self.__data, dtype=np.float)


    def __truediv__(self, other):
        if type(other) not in [int, float]:
            raise TypeError('Invalid scalar argument: {} (type={})'.format(other, type(other)))
        return Point(self.__data / other, dtype=np.float)


    def __iter__(self):
        for i in self.__data:
            yield i



class Rectangle:
    def __init__(self, *args, dtype=int):
        if dtype not in [int, float]:
            raise ValueError('dtype must be either int or float but {} is given'.format(dtype))

        # self.__data = np.array([left, top, right, bottom])
        if len(args) == 0:
            data = np.array([0, 0, -1, -1])
        elif len(args) == 1:
            data = np.array(args[0])
            if data.ndim != 1 or data.shape[0] != 4:
                raise ValueError('Invalid argument: {}'.format(args[0]))
        elif len(args) == 2:
            data = np.array([*args[0], *args[1]])
        elif len(args) == 4:
            data = np.array(args)
        else:
            raise ValueError('Invalid arguments: {}'.format(args))

        self.__data = data.astype(np.float) if dtype == float \
                else data.round().astype(np.int)

    @property
    def tl_corner(self):
        return Point(self.__data[0], self.__data[1], dtype=self.dtype)

    @tl_corner.setter
    def tl_corner(self, p):
        self.__data[0] = p.x
        self.__data[1] = p.y

    @property
    def tr_corner(self):
        return Point(self.__data[2], self.__data[1], dtype=self.dtype)

    @tr_corner.setter
    def tr_corner(self, p):
        self.__data[2] = p.x
        self.__data[1] = p.y

    @property
    def bl_corner(self):
        return Point(self.__data[0], self.__data[3], dtype=self.dtype)

    @bl_corner.setter
    def bl_corner(self, p):
        self.__data[0] = p.x
        self.__data[3] = p.y

    @property
    def br_corner(self):
        return Point(self.__data[2], self.__data[3], dtype=self.dtype)

    @br_corner.setter
    def br_corner(self, p):
        self.__data[2] = p.x
        self.__data[3] = p.y

    @property
    def left(self):
        return self.__data[0]

    @left.setter
    def left(self, val):
        self.__data[0] = val

    @property
    def top(self):
        return self.__data[1]

    @top.setter
    def top(self, val):
        self.__data[1] = val

    @property
    def right(self):
        return self.__data[2]

    @right.setter
    def right(self, val):
        self.__data[2] = val

    @property
    def bottom(self):
        return self.__data[3]

    @bottom.setter
    def bottom(self, val):
        self.__data[3] = val

    @property
    def dtype(self):
        return self.__data.dtype


    def astype(self, dtype):
        if dtype not in [int, float]:
            raise ValueError('dtype must be either int or float but {} is given'.format(dtype))
        return Rectangle(self.__data, dtype=dtype)


    def is_empty(self):
        return (self.top > self.bottom or self.left > self.right)


    def width(self):
        if self.is_empty():
            return 0
        else:
            return self.right - self.left + 1


    def height(self):
        if self.is_empty():
            return 0
        else:
            return self.bottom - self.top + 1


    def area(self):
        return self.width() * self.height()


    def center(self):
        return (self.tl_corner + self.br_corner) / 2


    def intersect(self, other):
        return Rectangle(
            max(self.left, other.left),
            max(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
            dtype=self.dtype
        )


    def to_patch(self, **kwargs):
        return patches.Rectangle(tuple(self.tl_corner),
                                 self.width() - 1, self.height() - 1,
                                 **kwargs)


    def to_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[self.top:self.bottom+1, self.left:self.right+1] = 255
        return mask


    def __repr__(self):
        return 'Rectangle(l={}, t={}, r={}, b={})'.format(*self.__data)


    def __str__(self):
        return 'Rectangle(l={}, t={}, r={}, b={})'.format(*self.__data)


    def __eq__(self, other):
        if isinstance(other, Rectangle):
            return np.array_equal(self.__data, other.__data)
        else: return False


    def __contains__(self, item):
        if isinstance(item, Point):
            return not (item.x < self.left or item.x > self.right or item.y < self.top or item.y > self.bottom)
        elif isinstance(item, Rectangle):
            return self.__add__(item) == self


    def __add__(self, other):
        if other.is_empty():
            return Rectangle(self.__data)
        elif self.is_empty():
            return Rectangle(other.__data)
        else:
            return Rectangle(
                min(self.left, other.left),
                min(self.top, other.top),
                max(self.right, other.right),
                max(self.bottom, other.bottom),
                dtype=self.dtype
            )


    def __iter__(self):
        xywh = (self.left, self.top, self.width(), self.height())
        for val in xywh:
            yield val


    def __le__(self, other):
        if self.left < other.left:
            return True
        elif self.left > other.left:
            return False

        elif self.top < other.top:
            return True
        elif self.top > other.top:
            return False

        elif self.right < other.right:
            return True
        elif self.right > other.right:
            return False

        elif self.bottom < other.bottom:
            return True
        elif self.bottom > other.bottom:
            return False

        else:
            return True



def get_rect(img):
    return Rectangle(0, 0, img.shape[1] - 1, img.shape[0] - 1)


def shrink_rect(rect: Rectangle, num):
    return Rectangle(rect.left+num, rect.top+num, rect.right-num, rect.bottom-num, dtype=rect.dtype)


def grow_rect(rect: Rectangle, num):
    return shrink_rect(rect, -num)


def translate_rect(rect: Rectangle, p: Point):
    return Rectangle(rect.left+p.x, rect.top+p.y, rect.right+p.x, rect.bottom+p.y, dtype=rect.dtype)



class Polygon(geom.Polygon):

    def degenerate(self):
        return not hasattr(self, 'exterior') or self.exterior is None

    def get_coordinates(self, shape=None):
        if self.degenerate():
            return np.array((), dtype=np.int32), np.array((), dtype=np.int32)
        else:
            x, y = self.exterior.xy
            x, y = np.round(x).astype(np.int32), np.round(y).astype(np.int32)
            if shape is not None:
                x, y = np.clip(x, 0, shape[1] - 1), np.clip(y, 0, shape[0] - 1)
            return x, y


    def to_ndarray(self, shape=None):
        x, y = self.get_coordinates(shape)
        return np.array(list(zip(x.tolist(), y.tolist())))


    def simplify(self, tolerance, preserve_topology=True):
        return Polygon(super(Polygon, self).simplify(tolerance, preserve_topology))


    def to_patch(self, shape, *args, **kwargs):
        return patches.Polygon(self.to_ndarray(), shape, *args, **kwargs)



def mask_to_polygons(mask):
    polys = []
    # Add padding to detect the contours at the border of the image
    contours = measure.find_contours(
        np.pad(
            # cv2.erode(
            #     mask,
            #     kernel=np.ones((3, 3), np.uint8)
            # ),
            mask,
            ((1, 1), (1, 1)), 'constant', constant_values=0
        ), 0.5, positive_orientation='low'
    )

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        fpoly = Polygon(contour).simplify(0.5, preserve_topology=False)
        if not fpoly.degenerate():
            polys.append(Polygon(fpoly.to_ndarray()))

    return polys


def polygons_to_mask(polys, shape):
    return cv2.erode(
        cv2.fillPoly(
            np.zeros(shape, dtype=np.uint8),
            [poly.to_ndarray(shape) for poly in polys],
            255
        ),
        kernel=np.ones((3, 3), np.uint8))
    # return cv2.fillPoly(
    #     np.zeros(shape, dtype=np.uint8),
    #     [poly.to_ndarray(shape) for poly in polys],
    #     255
    # )



def extract_bbox(poly_or_mask):
    if type(poly_or_mask) == Polygon:
        x, y = poly_or_mask.get_coordinates()
        return Rectangle(np.min(x), np.min(y), np.max(x), np.max(y))
    elif type(poly_or_mask) == np.ndarray:
        horizontal_indicies = np.where(np.any(poly_or_mask, axis=0))[0]
        vertical_indicies = np.where(np.any(poly_or_mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
        else:
            x1, x2, y1, y2 = 0, -1, 0, -1
        tl = Point(x1, y1)
        br = Point(x2, y2)
        return Rectangle(tl, br)
    else:
        raise TypeError('Unrecognizable argument type {}'.format(type(poly_or_mask)))


def extract_bbox_multi(poly_or_mask):
    r = Rectangle()
    for p in poly_or_mask:
        r += extract_bbox(p)
    return r