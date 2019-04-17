import enum
import cv2
import copy
import numpy as np
import matplotlib.patches as patches

from functools import partial
from matplotlib.widgets import Slider, RadioButtons

from .image_window import ImageWindow
from .geometry import Point
from .utils import hide_axes_labels, preprocess_mask, grabcut, overlay_mask, threshold
from .drag_interpreter import DragInterpreterBase


class ViewMode(enum.Enum):
    MASK = 0
    MASKED_IMAGE = 1
    INVERSE_MASKED_IMAGE = 2
    MASK_OVERLAY = 3


    def __add__(self, other: int):
        return ViewMode((self.value + other) % len(ViewMode))


    def __sub__(self, other: int):
        return self.__add__(-other)



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
            brush_type.value['val'] : np.array(brush_type.value['color'])
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
        super(BrushInterpreter, self).__init__()
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
        color = [val/255 for val in self.brush.value['color']]
        color.append(alpha)
        color = tuple(color)
        return patches.Circle(
            tuple(self.center), self.radius,
            edgecolor=color if alpha == 1.0 else 'k',
            facecolor=color if self.solid else 'none',
            linewidth=2
        )



def apply_brush_touch(img: np.ndarray, brush_touch: BrushTouch):
    key = 'val' if img.ndim == 2 else 'color'
    return cv2.circle(img, tuple(brush_touch.center), brush_touch.radius, brush_touch.brush.value[key], -1 if brush_touch.solid else 1)



class MaskModifyHistoryManager:

    def __init__(self):
        self.__action_history = []


    def __len__(self):
        return len(self.__action_history)


    def add_brush_touch_history(self, brush_trace):
        self.__action_history.append(('brush', copy.deepcopy(brush_trace)))


    def add_grabcut_history(self, mask):
        self.__action_history.append(('grabcut', np.copy(mask)))


    def add_thresh_history(self, lower_thresh, upper_thresh):
        self.__action_history.append(('thresh', (lower_thresh, upper_thresh)))


    def pop(self):
        if self.__len__() == 0:
            raise Exception('No history to pop')
        else:
            return self.__action_history.pop(-1)


    def brush_traces(self):
        return [h[1] for h in self.__action_history if h[0] == 'brush']



class MaskModifier(ImageWindow):
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
        self.src = np.copy(img)
        self.src_hsv = cv2.cvtColor(self.src, cv2.COLOR_RGB2HSV)
        self.mask_src = preprocess_mask(mask)
        self.mask = np.copy(self.mask_src)
        self.viewmode = ViewMode.MASK
        self.brush_iptr = BrushInterpreter()
        self.history_mgr = MaskModifyHistoryManager()
        self.save_result = False

        unit = 0.06
        self.brush_panel = self.fig.add_axes([unit, 0.9, len(BrushType)*unit, unit])
        hide_axes_labels(self.brush_panel)
        self.brush_panel.imshow(np.array([[BrushType.val2color(i) for i in range(len(BrushType))]]))
        self.brush_indicator = []
        self.update_brush_panel()

        self.fillhole_panel = self.fig.add_axes([(len(BrushType)+1)*unit, 0.9, unit, unit])
        hide_axes_labels(self.fillhole_panel)
        self.fillhole_panel.patch.set_facecolor('lightgoldenrodyellow')
        self.fillhole_panel.text(
            -0.45, -0.7,
            'Fill Holes',
            bbox=dict(
                linewidth=1,
                edgecolor='goldenrod',
                facecolor='none',
                alpha=1.0
            )
        )
        self.fillhole_panel.imshow(np.array([[[255, 255, 224]]], dtype=np.uint8))
        self.fillhole_switch = RadioButtons(self.fillhole_panel, ('on', 'off'))
        self.fillhole_switch.on_clicked(lambda x: self.update_mask() or self.display())

        self.upper_names = ['Upper H', 'Upper S', 'Upper V']
        self.lower_names = ['Lower H', 'Lower S', 'Lower V']
        self.min_values = np.array([0, 0, 0])
        self.max_values = np.array([179, 255, 255])

        self.sliders = {}
        self.slider_axes = []
        self.on_slider_adjust = False
        axcolor = 'lightgoldenrodyellow'
        for i in range(3):
            lower_slider_ax = self.fig.add_axes([0.1, 0.1 - (2 * i) * 0.015 - i * 0.005, 0.8, 0.01], facecolor=axcolor)
            self.sliders[self.lower_names[i]] = Slider(lower_slider_ax, self.lower_names[i], self.min_values[i], self.max_values[i], valinit=self.min_values[i])
            self.sliders[self.lower_names[i]].valtext.set_text(str(int(self.min_values[i])))
            self.sliders[self.lower_names[i]].on_changed(partial(self.slider_callback, i, 'lower'))
            self.slider_axes.append(lower_slider_ax)

            upper_slider_ax = self.fig.add_axes([0.1, 0.1 - (2 * i + 1) * 0.015 - i * 0.005, 0.8, 0.01], facecolor=axcolor)
            self.sliders[self.upper_names[i]] = Slider(upper_slider_ax, self.upper_names[i], self.min_values[i], self.max_values[i], valinit=self.max_values[i])
            self.sliders[self.upper_names[i]].valtext.set_text(str(int(self.max_values[i])))
            self.sliders[self.upper_names[i]].on_changed(partial(self.slider_callback, i, 'upper'))
            self.slider_axes.append(upper_slider_ax)
        self.display()


    def mainloop(self):
        super().mainloop()
        return self.mask if self.save_result else None


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


    def update_mask(self):
        thresh_mask = np.zeros(self.src.shape, dtype=np.uint8)
        for i in range(3):
            thresh_mask[:, :, i] = threshold(self.src_hsv[:, :, i], self.lower_thresh[i], self.upper_thresh[i])
        thresh_mask = np.amin(thresh_mask, axis=-1)

        if self.do_fillhole:
            h, w = thresh_mask.shape[:2]
            m = np.zeros((h + 2, w + 2), np.uint8)
            m_floodfill = thresh_mask.copy()
            cv2.floodFill(m_floodfill, m, (0,0), 255)
            thresh_mask = thresh_mask | cv2.bitwise_not(m_floodfill)

        thresh_mask //= 255
        self.mask = self.mask_src * thresh_mask

        for brush_trace in self.history_mgr.brush_traces():
            for brush_touch in brush_trace:
                self.mask = apply_brush_touch(self.mask, brush_touch)


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
    def do_fillhole(self):
        return self.fillhole_switch.value_selected == 'on'


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
            if self.lower_thresh[idx] > self.upper_thresh[idx]:
                self.sliders[other_name].set_val(val)


    def set_sliders(self, lower_thresh, upper_thresh):
        for i, vals in enumerate(zip(lower_thresh, upper_thresh)):
            self.sliders[self.lower_names[i]].set_val(vals[0])
            self.sliders[self.upper_names[i]].set_val(vals[1])
        self.update_mask()
        self.refresh()


    def display(self):
        if self.viewmode == ViewMode.MASK:
            rgb_mask = np.zeros((*self.mask.shape, 3), dtype=np.uint8)
            for val in range(4):
                rgb_mask[self.mask == val] = BrushType.val2color(val)
            self.set_image(rgb_mask)
            self.set_title('Object Mask')
        elif self.viewmode == ViewMode.MASKED_IMAGE:
            self.set_image(cv2.bitwise_and(self.src, self.src, mask=np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype(np.uint8)))
            self.set_title('Foreground')
        elif self.viewmode == ViewMode.INVERSE_MASKED_IMAGE:
            self.set_image(cv2.bitwise_and(self.src, self.src, mask=np.where((self.mask == 0) + (self.mask == 2), 255, 0).astype(np.uint8)))
            self.set_title('Background')
        elif self.viewmode == ViewMode.MASK_OVERLAY:
            overlayed = np.copy(self.src)
            for i in range(len(BrushType)):
                overlay_mask(overlayed, np.where(self.mask == i, 255, 0).astype(np.uint8), BrushType.val2color(i) / 255)
            self.set_image(overlayed)
            self.set_title('Mask Overlayed Image')
        else:
            raise ValueError('Invalid viewmode: {}'.format(self.viewmode))


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
                    self.set_sliders(*data)
                self.update_mask()
                self.display()
            else:
                self.show_message('No history to recover', 'Guide')


    def on_mouse_press(self, event):
        super().on_mouse_press(event)
        p = self.get_image_coordinates(event)
        if event.key == None and event.button == 3:
            self.brush_iptr.start_dragging(p)
        elif event.button == 1 and event.inaxes in self.slider_axes:
            self.on_slider_adjust = True
            self.history_mgr.add_thresh_history(self.lower_thresh, self.upper_thresh)


    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        p = self.get_image_coordinates(event)
        if p != None and event.key == None:
            self.add_transient_patch(BrushTouch(
                p, self.brush_iptr.radius, True, self.brush_iptr.brush
            ).patch(alpha=0.3))

            if self.brush_iptr.on_dragging:
                trace = self.brush_iptr.get_trace(p)
                self.add_patch(trace.patch())


    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        p = self.get_image_coordinates(event)
        if self.brush_iptr.on_dragging:
            if p != None:
                self.brush_iptr.get_trace(p)
                self.history_mgr.add_brush_touch_history(self.brush_iptr.history())
                self.brush_iptr.clear()
            self.clear_patches()
            self.clear_transient_patch()
            self.brush_iptr.finish_dragging(p)
            self.update_mask()
            self.display()
        elif self.on_slider_adjust:
            self.on_slider_adjust = False
            self.update_mask()
            self.display()


    def on_scroll(self, event):
        super().on_scroll(event)
        if event.key is None:
            if event.step == 1:
                self.brush_iptr.radius += 1
            elif self.brush_iptr.radius > 1:
                self.brush_iptr.radius -= 1
            self.refresh()

