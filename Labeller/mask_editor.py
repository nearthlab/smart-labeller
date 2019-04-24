import copy
import enum
from functools import partial

import cv2
import matplotlib.colors as colors
import numpy as np
from matplotlib.widgets import Slider, RadioButtons

from .brush import BrushType, BrushInterpreter, BrushTouch, apply_brush_touch
from .drag_interpreter import PolarDragInterpreter
from .geometry import Point
from .image_window import ImageWindow
from .utils import (hide_axes_labels, preprocess_mask,
                    grabcut, overlay_mask, threshold_hsv,
                    HRange, SRange, VRange, get_arc_regions,
                    fill_holes, largest_connected_component)


class ViewMode(enum.Enum):
    MASK = 0
    MASKED_IMAGE = 1
    INVERSE_MASKED_IMAGE = 2
    MASK_OVERLAY = 3

    def __add__(self, other: int):
        return ViewMode((self.value + other) % len(ViewMode))

    def __sub__(self, other: int):
        return self.__add__(-other)


class HSPlotMode(enum.Enum):
    ALL = 0
    IMAGE = 1
    FOREGROUND = 2
    BACKGROUND = 3

    def __add__(self, other: int):
        return HSPlotMode((self.value + other) % len(HSPlotMode))

    def __sub__(self, other: int):
        return self.__add__(-other)


class MaskEditHistoryManager:

    def __init__(self):
        self.last_lower_thresh = np.array([0, 0, 0])
        self.last_upper_thresh = np.array([359, 255, 255])
        self.__action_history = []

    def __len__(self):
        return len(self.__action_history)

    def __repr__(self):
        return '{} / {} / {}'.format(self.__action_history, self.last_lower_thresh, self.last_upper_thresh)

    def add_brush_touch_history(self, brush_trace):
        self.__action_history.append(('brush', copy.deepcopy(brush_trace)))

    def add_grabcut_history(self, mask):
        self.__action_history.append(('grabcut', np.copy(mask)))

    def add_thresh_history(self, lower_thresh, upper_thresh):
        self.__action_history.append(('thresh', (self.last_lower_thresh, self.last_upper_thresh)))
        self.last_lower_thresh = lower_thresh.copy()
        self.last_upper_thresh = upper_thresh.copy()

    def add_switch_history(self, name):
        self.__action_history.append((name, None))

    def pop(self):
        if self.__len__() == 0:
            raise Exception('No history to pop')
        else:
            item = self.__action_history.pop(-1)
            if item[0] == 'thresh':
                self.last_lower_thresh = item[1][0].copy()
                self.last_upper_thresh = item[1][1].copy()
            return item

    def brush_traces(self):
        return [h[1] for h in self.__action_history if h[0] == 'brush']


class MaskEditor(ImageWindow):
    '''
<Basic Actions>
a/d: switch to previous/next view mode
Ctrl + s: save and exit
Ctrl + z: undo the last action

<Brush Actions>
mouse right + dragging = paint with brush
mouse wheel up/down: increase/decrease brush radius
w/s: change brush type (current brush type is shown on the upper panel)

<Grabcut Actions>
Ctrl + g: run grabcut with current mask

<Threshold Actions>
Use sliders on the bottom to adjust thresholds for H, S, V channel pixel values
    '''

    def __init__(self, img: np.ndarray, mask: np.ndarray, win_title=None):
        super().__init__(win_title, (0.05, 0.18, 0.9, 0.7))
        self.disable_callbacks()

        self.src = np.copy(img)
        self.src_hsv = cv2.cvtColor(self.src, cv2.COLOR_RGB2HSV)
        self.mask_src = preprocess_mask(mask)
        self.mask = np.copy(self.mask_src)
        self.viewmode = ViewMode.MASK
        self.brush_iptr = BrushInterpreter()
        self.history_mgr = MaskEditHistoryManager()
        self.save_result = False

        self.pixel_panel = self.fig.add_axes((0.7, 0.9, 0.08, 0.05))
        self.pixel_panel.imshow(255 * np.ones((5, 8, 3), np.uint8))
        hide_axes_labels(self.pixel_panel)
        for pos in ['left', 'top', 'right', 'bottom']:
            self.pixel_panel.spines[pos].set_color('none')

        unit = 0.06
        # Create brush panel
        self.brush_panel = self.fig.add_axes((unit, 0.9, len(BrushType) * unit, unit))
        hide_axes_labels(self.brush_panel)
        self.brush_panel.imshow(np.array([[BrushType.val2color(i) for i in range(len(BrushType))]]))
        self.brush_indicator = []
        self.update_brush_panel()

        # Create largest component panel
        self.lc_panel = self.fig.add_axes(((len(BrushType) + 1) * unit, 0.9, unit, unit))
        hide_axes_labels(self.lc_panel)
        self.lc_panel.text(
            -0.45, -0.7,
            'Largest Component Only',
            bbox=dict(
                linewidth=1,
                edgecolor='goldenrod',
                facecolor='none',
                alpha=1.0
            )
        )
        self.lc_panel.imshow(np.array([[[255, 255, 224]]], dtype=np.uint8))
        self.lc_switch = RadioButtons(self.lc_panel, ('off', 'on'))
        self.lc_switch.on_clicked(lambda x: self.update_mask() or self.history_mgr.add_switch_history('lc') or self.display())

        # Create threshold sliders
        self.upper_names = ['Upper H', 'Upper S', 'Upper V']
        self.lower_names = ['Lower H', 'Lower S', 'Lower V']
        self.min_values = np.array([0, 0, 0])
        self.max_values = np.array([359, 255, 255])
        self.sliders = {}
        self.slider_axes = []
        self.on_slider_adjust = False
        axcolor = 'lightgoldenrodyellow'
        for i in range(3):
            lower_slider_ax = self.fig.add_axes((0.25, 0.12 - (2 * i) * 0.018 - i * 0.008, 0.7, 0.01), facecolor=axcolor)
            self.sliders[self.lower_names[i]] = Slider(lower_slider_ax, self.lower_names[i], self.min_values[i], self.max_values[i], valinit=self.min_values[i], color=(0, 1, 0, 0.3))
            self.sliders[self.lower_names[i]].valtext.set_text(str(int(self.min_values[i])))
            self.slider_axes.append(lower_slider_ax)

            upper_slider_ax = self.fig.add_axes((0.25, 0.12 - (2 * i + 1) * 0.018 - i * 0.008, 0.7, 0.01), facecolor=axcolor)
            self.sliders[self.upper_names[i]] = Slider(upper_slider_ax, self.upper_names[i], self.min_values[i], self.max_values[i], valinit=self.max_values[i], color=(0, 1, 0, 0.3))
            self.sliders[self.upper_names[i]].valtext.set_text(str(int(self.max_values[i])))
            self.slider_axes.append(upper_slider_ax)
        self.enable_callbacks()

        self.hs_panel = self.fig.add_axes((0.008, 0.01, 0.15, 0.15), projection='polar', facecolor='lightgoldenrodyellow')
        self.hs_plot_mode = HSPlotMode.ALL
        self.hs_region_iptr = PolarDragInterpreter()
        self.arc_regions = []
        self.temp_arc_regions = []
        hide_axes_labels(self.hs_panel)

        self.v_panel = self.fig.add_axes((0.16, 0.01, 0.02, 0.15), facecolor='lightgoldenrodyellow')
        hide_axes_labels(self.v_panel, 'x')

        self.plot_hs_range()
        self.plot_thresh_regions()
        self.display()

    def mainloop(self):
        super().mainloop()
        return self.mask if self.save_result else None

    def enable_callbacks(self):
        super().enable_callbacks()
        if hasattr(self, 'sliders'):
            for i in range(3):
                self.sliders[self.lower_names[i]].set_active(True)
                self.cids.append(self.sliders[self.lower_names[i]].on_changed(partial(self.slider_callback, i, 'lower')))
                self.sliders[self.upper_names[i]].set_active(True)
                self.cids.append(self.sliders[self.upper_names[i]].on_changed(partial(self.slider_callback, i, 'upper')))

    def disable_callbacks(self):
        super().disable_callbacks()
        if hasattr(self, 'sliders'):
            for i in range(3):
                self.sliders[self.lower_names[i]].set_active(False)
                self.sliders[self.upper_names[i]].set_active(False)

    def run_grabcut(self):
        gc_mask = grabcut(self.src, cv2.GC_INIT_WITH_MASK, mask=self.mask)
        self.mask, gc_mask = gc_mask, self.mask
        self.display()
        self.history_mgr.add_grabcut_history(self.mask_src)
        self.mask_src = gc_mask
        self.update_mask()
        self.display()

    @property
    def lower_thresh(self):
        return [self.sliders[self.lower_names[i]].val for i in range(3)]

    @property
    def upper_thresh(self):
        return [self.sliders[self.upper_names[i]].val for i in range(3)]

    @property
    def h_range(self):
        return HRange(self.lower_thresh[0], self.upper_thresh[0])

    @property
    def s_range(self):
        return SRange(self.lower_thresh[1], self.upper_thresh[1])

    @property
    def v_range(self):
        return VRange(self.lower_thresh[2], self.upper_thresh[2])

    def update_mask(self):
        # thresh_mask = np.zeros(self.src.shape, dtype=np.uint8)
        # for i in range(3):
        #     thresh_mask[:, :, i] = threshold(self.src_hsv[:, :, i], self.lower_thresh[i], self.upper_thresh[i])
        # thresh_mask = np.amin(thresh_mask, axis=-1) // 255
        thresh_mask = threshold_hsv(self.src_hsv, self.h_range, self.s_range, self.v_range) // 255

        self.mask = self.mask_src * thresh_mask

        for brush_trace in self.history_mgr.brush_traces():
            for brush_touch in brush_trace:
                self.mask = apply_brush_touch(self.mask, brush_touch)

        if self.largest_component_only:
            lc_mask = largest_connected_component(
                np.where(self.mask % 2 == 1, 255, 0).astype(np.uint8)
            ) // 255
            #                   lc_mask|   0   1
            #         self.mask        |
            # --------------------------|--------
            #           0(BG)          |   0   0
            #           1(FG)          |   0   1
            #           2(PBG)         |   2   2
            #           3(PFG)         |   2   3
            q, r = self.mask.__divmod__(2)
            self.mask = 2 * q + lc_mask * r

        # Most of the semantic/instance segmentation datasets require
        # object masks to be simply connected (i.e. contains no holes)
        # So fill the holes in final (probably) foreground mask
        fg = np.where(self.mask == 1, 255, 0).astype(np.uint8)
        pfg = np.where(self.mask == 3, 255, 0).astype(np.uint8)
        filled_fg = fill_holes(fg)
        filled_pfg = fill_holes(pfg)
        self.mask[filled_fg == 255] = 1
        self.mask[filled_pfg == 255] = 3

    def update_brush_panel(self):
        for item in self.brush_indicator:
            item.remove()
        self.brush_indicator.clear()
        brush_id = self.brush_iptr.brush.value['val']
        self.brush_indicator = [
            self.brush_panel.text(
                i - 0.45, -0.7,
                BrushType.val2name(i),
                fontweight='bold' if brush_id == i else 'normal',
                bbox=dict(
                    linewidth=3 if brush_id == i else 1,
                    facecolor='w',
                    edgecolor=BrushType.val2color(i) / 255,
                    alpha=1.0
                )
            ) for i in range(len(BrushType))
        ]
        self.refresh()

    @property
    def largest_component_only(self):
        return self.lc_switch.value_selected == 'on'

    def slider_callback(self, idx, pos, val):
        assert pos in ['upper', 'lower']
        name, other_name = (self.upper_names[idx], self.lower_names[idx]) if pos == 'upper' \
            else (self.lower_names[idx], self.upper_names[idx])

        # This is to convert val to native Python type
        # to avoid val being an instance of np.float or np.int
        # which causes an unexpected behaviour sometimes
        if hasattr(val, 'item'):
            val = val.item()

        if isinstance(val, float):
            self.sliders[name].set_val(int(round(val)))
        elif isinstance(val, int):
            self.sliders[name].valtext.set_text(val)

    def set_sliders(self, lower_thresh, upper_thresh, write_history=True):
        for i, vals in enumerate(zip(lower_thresh, upper_thresh)):
            self.sliders[self.lower_names[i]].set_val(vals[0])
            self.sliders[self.upper_names[i]].set_val(vals[1])
        if write_history:
            self.history_mgr.add_thresh_history(self.lower_thresh, self.upper_thresh)
        self.update_mask()
        self.refresh()

    def display(self):
        if self.viewmode == ViewMode.MASK:
            rgb_mask = np.zeros((*self.mask.shape, 3), dtype=np.uint8)
            for val in range(4):
                rgb_mask[self.mask == val] = BrushType.val2color(val)
            self.set_image(rgb_mask)
            self.ax.set_title('Object Mask')
        elif self.viewmode == ViewMode.MASKED_IMAGE:
            self.set_image(cv2.bitwise_and(self.src, self.src, mask=np.where(self.mask % 2 == 1, 255, 0).astype(np.uint8)))
            self.ax.set_title('Foreground')
        elif self.viewmode == ViewMode.INVERSE_MASKED_IMAGE:
            self.set_image(cv2.bitwise_and(self.src, self.src, mask=np.where(self.mask % 2 == 0, 255, 0).astype(np.uint8)))
            self.ax.set_title('Background')
        elif self.viewmode == ViewMode.MASK_OVERLAY:
            overlayed = np.copy(self.src)
            for i in range(len(BrushType)):
                overlay_mask(overlayed, np.where(self.mask == i, 255, 0).astype(np.uint8), BrushType.val2color(i) / 255)
            self.set_image(overlayed)
            self.ax.set_title('Mask Overlayed Image')
        else:
            raise ValueError('Invalid viewmode: {}'.format(self.viewmode))

    def add_arc_region(self, rect, temporary=False):
        if not rect.is_empty():
            theta = np.arange(rect.left, rect.right, np.pi / 180)
            r_bottom = rect.bottom * np.ones_like(theta)
            r_top = rect.top * np.ones_like(theta)

            whole_theta_range = (rect.right - rect.left) % (2 * np.pi) == 0

            if temporary:
                if not whole_theta_range:
                    self.temp_arc_regions += self.hs_panel.plot(
                        [rect.left, rect.left], [rect.bottom, rect.top], color='k', linestyle='--'
                    )
                    self.temp_arc_regions += self.hs_panel.plot(
                        [rect.right, rect.right], [rect.bottom, rect.top], color='k', linestyle='--'
                    )

                self.temp_arc_regions += self.hs_panel.plot(theta, r_top, color='k', linestyle='--')
                self.temp_arc_regions += self.hs_panel.plot(theta, r_bottom, color='k', linestyle='--')
            else:
                if not whole_theta_range:
                    self.arc_regions += self.hs_panel.plot(
                        [rect.left, rect.left], [rect.bottom, rect.top], color='k', linestyle='-'
                    )
                    self.arc_regions += self.hs_panel.plot(
                        [rect.right, rect.right], [rect.bottom, rect.top], color='k', linestyle='-'
                    )
                self.arc_regions += self.hs_panel.plot(theta, r_top, color='k', linestyle='-')
                self.arc_regions += self.hs_panel.plot(theta, r_bottom, color='k', linestyle='-')

            self.hs_panel.set_rmax(1.2)
            self.refresh()

    def clear_arc_regions(self):
        for i in range(len(self.arc_regions)):
            self.arc_regions[i].remove()

        self.arc_regions.clear()
        self.refresh()

    def clear_temp_arc_regions(self):
        for i in range(len(self.temp_arc_regions)):
            self.temp_arc_regions[i].remove()

        self.temp_arc_regions.clear()
        self.refresh()

    def plot_hs_range(self):
        self.hs_panel.clear()
        if self.hs_plot_mode == HSPlotMode.ALL:
            self.hs_panel.set_title('Hue-Saturation\n        Disc', loc='left')
            # visualize hsv pallete
            h_range = HRange()
            s_range = SRange()

            r_val = s_range.get_ranges(1 / 32)[0]
            theta_val = h_range.get_ranges(np.pi / 60)[0]

            r = np.tile(r_val, len(theta_val))
            theta = np.repeat(theta_val, len(r_val))
            self.hs_panel.scatter(
                theta, r,
                c=colors.hsv_to_rgb(np.clip(np.transpose([theta / np.pi, r, np.ones_like(theta)]), 0, 1)),
                s=30 * r,
                alpha=0.8
            )
        else:
            if self.hs_plot_mode == HSPlotMode.IMAGE:
                img = np.copy(self.src_hsv)
                self.hs_panel.set_title('Image HSV\nDistribution', loc='left')
            elif self.hs_plot_mode == HSPlotMode.BACKGROUND:
                img = cv2.bitwise_and(self.src_hsv, self.src_hsv, mask=np.where(self.mask % 2 == 0, 255, 0).astype(np.uint8))
                self.hs_panel.set_title('Background HSV\nDistribution', loc='left')
            else:
                img = cv2.bitwise_and(self.src_hsv, self.src_hsv, mask=np.where(self.mask % 2 == 1, 255, 0).astype(np.uint8))
                self.hs_panel.set_title('Foreground HSV\nDistribution', loc='left')
            hsv_pixels = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
            hsv_pixels = np.unique(hsv_pixels, axis=0)
            hsv_pixels = (np.round(hsv_pixels / [3, 8, 1]) * np.array([3, 8, 1])).astype(np.int)
            hsv_pixels = np.unique(hsv_pixels, axis=0)

            theta = hsv_pixels[:, 0] / 179 * np.pi
            r = hsv_pixels[:, 1] / 255
            v = hsv_pixels[:, 2] / 255
            self.hs_panel.scatter(
                theta, r,
                c=colors.hsv_to_rgb(np.clip(np.transpose([theta / np.pi, r, v]), 0, 1)),
                s=30 * r,
                alpha=0.8
            )

        self.hs_panel.set_rmax(1.2)

    def plot_thresh_regions(self):
        self.clear_arc_regions()
        for region in get_arc_regions(self.h_range, self.s_range):
            self.add_arc_region(region)
        self.hs_panel.set_rmax(1.2)

        self.v_panel.clear()
        rgb = cv2.cvtColor(np.array([[[0, 0, v]] * 10 for v in range(VRange.MAX + 1)]).astype(np.uint8), cv2.COLOR_HSV2RGB)
        alpha = np.array([[[255 if v in self.v_range else 0]] * 10 for v in range(VRange.MAX + 1)]).astype(np.uint8)
        self.v_panel.imshow(np.dstack((rgb, alpha)) / 255)

    def on_key_press(self, event):
        super().on_key_press(event)
        if event.key == 'd':
            self.viewmode += 1
            self.display()
        elif event.key == 'a':
            self.viewmode -= 1
            self.display()
        elif event.key == 'w':
            self.brush_iptr.brush += 1
            self.update_brush_panel()
        elif event.key == 's':
            self.brush_iptr.brush -= 1
            self.update_brush_panel()
        elif event.key == 'ctrl+s':
            self.save_result = len(self.history_mgr) > 0
            self.close()
        elif event.key == 'ctrl+g':
            self.run_grabcut()
        elif event.key == 'ctrl+z':
            if len(self.history_mgr) > 0:
                action_name, data = self.history_mgr.pop()
                if action_name == 'grabcut':
                    self.mask_src = data
                elif action_name == 'thresh':
                    self.set_sliders(*data, False)
                    self.plot_thresh_regions()
                elif action_name == 'lc':
                    idx = 0 if self.lc_switch.value_selected == 'off' else 1
                    self.lc_switch.set_active(1 - idx)
                    self.history_mgr.pop()
                self.update_mask()
                self.display()
            else:
                self.show_message('No history to recover', 'Guide')
        elif event.key == 'q':
            self.hs_plot_mode -= 1
            self.plot_hs_range()
            self.plot_thresh_regions()
        elif event.key == 'e':
            self.hs_plot_mode += 1
            self.plot_hs_range()
            self.plot_thresh_regions()

    def on_mouse_press(self, event):
        super().on_mouse_press(event)

        p = self.get_axes_coordinates(event)
        if event.key is None and event.button == 3 and event.inaxes == self.ax:
            self.brush_iptr.start_dragging(p)
        elif event.button == 1 and event.inaxes in self.slider_axes:
            self.clear_arc_regions()
            self.on_slider_adjust = True
        elif event.key in [None, 'shift', 'control', 'ctrl+shift'] and event.inaxes == self.hs_panel:
            if event.dblclick and event.button == 1:
                if not np.array_equal(self.history_mgr.last_lower_thresh, self.min_values) or not np.array_equal(self.history_mgr.last_upper_thresh, self.max_values):
                    self.clear_arc_regions()
                    self.set_sliders(self.min_values, self.max_values)
                    self.plot_thresh_regions()
                    self.update_mask()
                    self.display()
            elif event.button == 3:
                self.clear_arc_regions()
                p = self.get_axes_coordinates(event, dtype=float)
                if event.key in ['control', 'ctrl+shift'] and p is not None:
                    self.hs_region_iptr.start_dragging(Point(p.x, 0, dtype=float), event.key == 'control')
                else:
                    self.hs_region_iptr.start_dragging(p, event.key is None)

    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        p = self.get_axes_coordinates(event)
        if event.inaxes == self.ax and p in self.img_rect:
            self.transient_patches.append(
                self.pixel_panel.text(
                    0, 5,
                    'x: {}, y: {}, H: {}, S: {}, V: {}'
                        .format(p.x, p.y, self.src_hsv[p.y][p.x][0], self.src_hsv[p.y][p.x][1], self.src_hsv[p.y][p.x][2]),
                    bbox=dict(
                        linewidth=1,
                        edgecolor='none',
                        facecolor='none',
                        alpha=1.0
                    )
                )
            )
        if event.key is None and event.inaxes == self.ax:
            self.add_transient_patch(BrushTouch(
                p, self.brush_iptr.radius, True, self.brush_iptr.brush
            ).patch(alpha=0.3))

            if self.brush_iptr.on_dragging:
                trace = self.brush_iptr.get_trace(p)
                self.add_patch(trace.patch())
        elif self.on_slider_adjust:
            self.plot_thresh_regions()
        elif self.hs_region_iptr.on_dragging:
            self.clear_temp_arc_regions()
            p = self.get_axes_coordinates(event, dtype=float)
            self.hs_region_iptr.update(p, event.key is None)
            self.add_arc_region(self.hs_region_iptr.rect, temporary=True)

    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        p = self.get_axes_coordinates(event)
        if self.brush_iptr.on_dragging and event.button == 3:
            if p is not None:
                self.brush_iptr.get_trace(p)
                self.history_mgr.add_brush_touch_history(self.brush_iptr.history())
                self.brush_iptr.clear()
            self.clear_patches()
            self.clear_transient_patch()
            self.brush_iptr.finish_dragging(p)
            self.update_mask()
            self.display()
        elif self.on_slider_adjust and event.button == 1:
            self.on_slider_adjust = False
            self.history_mgr.add_thresh_history(self.lower_thresh, self.upper_thresh)
            self.update_mask()
            self.plot_thresh_regions()
            self.display()
        elif self.hs_region_iptr.on_dragging:
            p = self.get_axes_coordinates(event, dtype=float)
            self.hs_region_iptr.finish_dragging(p, event.key is None)
            self.clear_temp_arc_regions()

            rect = self.hs_region_iptr.rect
            if not rect.is_empty():
                hmin = int(round(np.rad2deg(rect.left).item())) % 360
                hmax = int(round(np.rad2deg(rect.right).item()))
                lower_thresh = [hmin, int(round(255 * rect.top)), self.v_range.min]
                upper_thresh = [hmax, int(round(255 * rect.bottom)), self.v_range.max]
                self.set_sliders(lower_thresh, upper_thresh)
                self.update_mask()
                self.display()

            self.plot_thresh_regions()

    def on_scroll(self, event):
        super().on_scroll(event)
        if event.key is None:
            if event.step == 1:
                self.brush_iptr.radius += 1
            elif self.brush_iptr.radius > 1:
                self.brush_iptr.radius -= 1
            self.refresh()
