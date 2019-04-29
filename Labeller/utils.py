import colorsys
import random
import subprocess
import warnings

import cv2
import numpy as np

from .geometry import Point, Rectangle, extract_bbox


def caps_lock_status():
    # Check if caps lock is on
    result = str(subprocess.check_output('xset -q | grep Caps', shell=True))
    return 'on' in result[result.find('00:'):result.find('01:')]


def on_caps_lock_off(func):
    def wrapper(*args, **kwargs):
        if not caps_lock_status():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def hide_axes_labels(axes, hide='xy'):
    if 'x' in hide:
        axes.set_xticklabels([])
        axes.get_xaxis().set_visible(False)

    if 'y' in hide:
        axes.set_yticklabels([])
        axes.get_yaxis().set_visible(False)


def preprocess_mask(mask):
    bbox = extract_bbox(mask)
    # make the mask to have only probably BG/FG values
    prob_mask = mask // 255 + 2
    # pixels outside of the bounding box are surely BG
    prob_mask[bbox.to_mask(prob_mask.shape) == 0] = 0

    return prob_mask


def merge_gc_mask(gc_mask, mask):
    #                     mask |   0   1
    #  gc_mask                 |
    # -------------------------|--------
    #   0(BG)                  |   0   0
    #   1(FG)                  |   0   1
    #   2(PBG)                 |   2   2
    #   3(PFG)                 |   2   3
    q, r = gc_mask.__divmod__(2)
    return 2 * q + mask * r


def grabcut(img, mode, mask=None, rect=None):
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    if mode == cv2.GC_INIT_WITH_RECT:
        if rect is None:
            raise ValueError('Grabcut initial rectangle is not provided')
        elif rect.is_empty():
            raise ValueError('Grabcut initial rectangle is empty')
    elif mode == cv2.GC_INIT_WITH_MASK:
        if mask is None:
            raise ValueError('Grabcut initial mask is not provided')
        elif np.array_equal(mask, np.zeros_like(mask)):
            raise ValueError('Grabcut initial mask contains no foreground pixels')
    else:
        raise ValueError('mode must be one of {} but {} is provided'
                         .format([cv2.GC_INIT_WITH_RECT, cv2.GC_INIT_WITH_MASK], mode))

    rect = Rectangle() if rect is None else rect
    mask = np.zeros(img.shape[:2], dtype=np.uint8) if mask is None else mask

    cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)

    return mask


def random_colors(N, bright=True, seed=4, uint8=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # Use random.Random(*) to produce the same sequence of random colors every time
    random.Random(seed).shuffle(colors)
    if uint8:
        return np.clip(np.array(colors) * 255, 0, 255).astype(np.uint8)
    else:
        return colors


def overlay_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class Range:
    MIN = 0
    MAX = 1
    NORM_RANGE = 1

    def __init__(self, minval=None, maxval=None):
        minval = minval or self.__class__.MIN
        maxval = maxval or self.__class__.MAX

        self.__inside = minval <= maxval
        if not self.__inside:
            minval, maxval = maxval, minval
        self.__min = max(self.__class__.MIN, minval)
        self.__max = min(self.__class__.MAX, maxval)

    @property
    def min(self):
        return self.__min

    @property
    def max(self):
        return self.__max

    @property
    def inside(self):
        return self.__inside

    def __contains__(self, item):
        if self.__inside:
            return item in range(self.__min, self.__max + 1)
        else:
            return item in range(self.__min + 1) or item in range(self.__max, self.__class__.MAX + 1)

    def __repr__(self):
        if self.__inside:
            return '[{}, {}]'.format(self.__min, self.__max)
        else:
            return '[{}, {}] U [{}, {}]'.format(self.__class__.MIN, self.__min, self.__max, self.__class__.MAX)

    def get_ranges(self, step):
        return [np.arange(self.__min / self.__class__.MAX * self.__class__.NORM_RANGE, (self.__max + 1) / self.__class__.MAX * self.__class__.NORM_RANGE, step)] if self.__inside \
            else [np.arange(self.__class__.MIN / self.__class__.MAX * self.__class__.NORM_RANGE, (self.__min + 1) / self.__class__.MAX * self.__class__.NORM_RANGE, step),
                  np.arange(self.__max / self.__class__.MAX * self.__class__.NORM_RANGE, (self.__class__.MAX + 1) / self.__class__.MAX * self.__class__.NORM_RANGE, step)]

    def __truediv__(self, other):
        if self.__inside:
            return self.__class__(self.__min // other, self.__max // other)
        else:
            return self.__class__(self.__max // other, self.__min // other)


class HRange(Range):
    MIN = 0
    MAX = 359
    NORM_RANGE = 2 * np.pi


class SRange(Range):
    MIN = 0
    MAX = 255


class VRange(Range):
    MIN = 0
    MAX = 255


def threshold(gray, range: Range):
    output = np.zeros_like(gray, np.uint8)
    if range.inside:
        output[np.bitwise_and(range.min <= gray, gray <= range.max)] = 255
    else:
        output[np.bitwise_or(range.min >= gray, gray >= range.max)] = 255

    return output


def threshold_hsv(hsv, h_range: HRange, s_range: SRange, v_range: VRange):
    output = np.zeros_like(hsv, np.uint8)
    output[:, :, 0] = threshold(hsv[:, :, 0], h_range / 2)
    output[:, :, 1] = threshold(hsv[:, :, 1], s_range)
    output[:, :, 2] = threshold(hsv[:, :, 2], v_range)

    return np.amin(output, axis=-1)


def get_arc_regions(h_range: HRange, s_range: SRange):
    hmin, hmax = (h_range.min, h_range.max) if h_range.inside else (h_range.max - HRange.MAX, h_range.min)
    return [
        Rectangle(
            hmin / 180 * np.pi, s_range.min / 256,
            (hmax + 1) / 180 * np.pi, (s_range.max + 1) / 256,
            dtype=float
        )
    ] if s_range.inside else [
        Rectangle(
            hmin / 180 * np.pi, 0,
            (hmax + 1) / 180 * np.pi, (s_range.min + 1) / 256,
            dtype=float
        ),
        Rectangle(
            hmin / 180 * np.pi, s_range.max / 256,
            (hmax + 1) / 180 * np.pi, 1,
            dtype=float
        )
    ]


def fill_holes(mask):
    h, w = mask.shape[:2]
    m = np.zeros((h + 4, w + 4), np.uint8)
    flood_filled = np.pad(
        mask, ((1, 1), (1, 1)),
        mode='constant', constant_values=0
    )
    cv2.floodFill(flood_filled, m, (0, 0), 255)
    return mask | cv2.bitwise_not(flood_filled[1:h + 1, 1:w + 1])


def fill_holes_gc(gc_mask):
    filled = gc_mask.copy()
    fg = np.where(filled == 1, 255, 0).astype(np.uint8)
    pfg = np.where(filled == 3, 255, 0).astype(np.uint8)
    filled_fg = fill_holes(fg)
    filled_pfg = fill_holes(pfg)
    filled[filled_fg == 255] = 1
    filled[filled_pfg == 255] = 3

    return filled


class ConnectedComponents:
    def __init__(self, mask: np.ndarray):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        self.__idx = sorted(list(range(1, num_labels)), key=lambda i: stats[i][cv2.CC_STAT_AREA], reverse=True)
        self.__labels = np.array(labels)
        self.__stats = np.array(stats)
        self.__centroids = np.array(centroids)

    def __len__(self):
        return len(self.__idx)

    def area(self, i):
        return self.__stats[self.__idx[i]][cv2.CC_STAT_AREA]

    def rect(self, i):
        l, t, w, h, _ = self.__stats[self.__idx[i]]
        return Rectangle(
            l, t, l + w - 1, t + h - 1
        )

    def centroid(self, i):
        return Point(*self.__centroids[self.__idx[i]], dtype=float)

    def mask(self, i):
        return np.where(self.__labels == self.__idx[i], 255, 0)

    def background(self):
        return np.where(self.__labels == 0, 255, 0)


def largest_connected_component(mask: np.ndarray):
    connected_components = ConnectedComponents(mask)
    if len(connected_components) == 0:
        warnings.warn('Couldn\'t find any connected component in the foreground')
        return np.ones_like(mask)
    else:
        return connected_components.mask(0)


def filter_by_area(mask: np.ndarray, area_ratio_thresh: float):
    connected_components = ConnectedComponents(mask)

    if len(connected_components) > 0:
        area_thresh = connected_components.area(0) * area_ratio_thresh
        j = 0
        for i in reversed(range(len(connected_components))):
            if connected_components.area(i) > area_thresh:
                j = i
                break
            else:
                continue

        filtered = np.zeros_like(mask)
        for i in range(j + 1):
            filtered = np.maximum(filtered, connected_components.mask(i))

        return filtered
    else:
        warnings.warn('Couldn\'t find any connected component in the foreground')
        return np.ones_like(mask)
