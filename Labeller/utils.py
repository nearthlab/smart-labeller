import colorsys
import random
import cv2
import numpy as np

from .geometry import Rectangle, extract_bbox


def hide_axes_labels(axes):
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)


def preprocess_mask(mask):
    bbox = extract_bbox(mask)
    # make the mask to have only probably BG/FG values
    prob_mask = mask // 255 + 2
    # pixels outside of the bounding box are surely BG
    prob_mask[bbox.to_mask(prob_mask.shape) == 0] = 0
    return prob_mask


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


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # Use random.Random(*) to produce the same sequence of random colors every time
    random.Random(4).shuffle(colors)
    return colors


def overlay_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def threshold(src, lb, ub):
    _, lb_mask = cv2.threshold(src, lb - 1, 255, cv2.THRESH_BINARY)
    _, ub_mask = cv2.threshold(src, ub + 1, 255, cv2.THRESH_BINARY_INV)
    return np.minimum(lb_mask, ub_mask)
