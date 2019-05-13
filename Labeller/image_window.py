import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .drag_interpreter import DragInterpreter
from .geometry import Point, Rectangle, get_rect, grow_rect, translate_rect
from .popups import MessageBox, MultipleChoiceQuestionAsker, YesNoQuestionAsker
from .utils import caps_lock_status, on_caps_lock_off


class ImageWindow(object):
    '''
F1: Displays this message
ESC: Close the current window

<Image Actions>
F5: Reset the image view
Ctrl + mouse left + dragging: translating image
Ctrl + mouse right + dragging: zoom in to the region you've specified
Ctrl + mouse wheel up/down: zoom in/out
    '''
    # [left, bottom, width, height]
    DEFAULT_AXES_POSITION = (0.05, 0.05, 0.9, 0.9)

    @classmethod
    def documentation(cls):
        s = ''
        for base in cls.__bases__:
            if hasattr(base, 'documentation'):
                s += base.documentation()
        if cls.__doc__ is not None:
            s += cls.__doc__
        return s

    def __init__(self, win_title=None, axes_pos=DEFAULT_AXES_POSITION):
        self.verbose = False
        self.root = tk.Tk()
        self.root.title(win_title or self.__class__.__name__)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        width = 5 * self.root.winfo_screenwidth() // 9
        height = self.root.winfo_screenheight()
        x = self.root.winfo_screenwidth() // 3
        y = 0
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        self.fig, self.ax = plt.subplots()

        self.fig.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.fig.canvas.draw()
        self.fig.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ax.set_xlabel('col')
        self.ax.set_ylabel('row')
        self.ax.set_facecolor('lightgoldenrodyellow')
        self.ax.set_position(axes_pos)

        self.cids = []
        self.enable_callbacks()

        self.img_rect = Rectangle()
        self.scope = Rectangle(0.0, 0.0, 1.0, 1.0, dtype=float)
        self.panning_iptr = DragInterpreter()
        self.zoom_iptr = DragInterpreter()
        self.patches = []
        self.transient_patches = []

    def refresh(self):
        self.fig.canvas.draw_idle()

    def force_focus(self):
        self.fig.canvas.get_tk_widget().focus_force()

    def set_image(self, img: np.ndarray):
        self.img_rect = get_rect(img)
        self.ax.clear()
        self.ax.imshow(img)
        self.adjust_view()

    def set_title(self, title: str):
        self.root.title(title)

    def adjust_view(self):
        r = self.roi()
        r = grow_rect(r, np.sqrt(r.area()) * 0.04)
        self.ax.set_xlim(
            left=r.left,
            right=r.right + 1
        )
        self.ax.set_ylim(
            top=r.top,
            bottom=r.bottom + 1
        )
        self.refresh()

    def get_scope(self, rect: Rectangle):
        return Rectangle(
            rect.left / self.img_rect.width(),
            rect.top / self.img_rect.height(),
            rect.right / self.img_rect.width(),
            rect.bottom / self.img_rect.height(),
            dtype=float
        )

    def roi(self):
        subr = (
            (1 - self.scope.left) * self.img_rect.left + self.scope.left * self.img_rect.right,
            (1 - self.scope.top) * self.img_rect.top + self.scope.top * self.img_rect.bottom,
            (1 - self.scope.right) * self.img_rect.left + self.scope.right * self.img_rect.right,
            (1 - self.scope.bottom) * self.img_rect.top + self.scope.bottom * self.img_rect.bottom
        )
        return Rectangle(*subr)

    def grow_scope(self, p: Point):
        c = self.scope.center()
        p = Point(p.x / self.img_rect.width(), p.y / self.img_rect.height(), dtype=float)

        delta_tl = (self.scope.tl_corner - c)
        tl = c + (delta_tl.length() + 0.01) * delta_tl.normalize()

        delta_br = (self.scope.br_corner - c)
        br = c + (delta_br.length() + 0.01) * delta_br.normalize()

        self.scope = translate_rect(Rectangle(tl.x, tl.y, br.x, br.y, dtype=float), (p - c) / 10)

    def shrink_scope(self, p: Point):
        c = self.scope.center()
        eps = 1 / Point(self.img_rect.width(), self.img_rect.height()).length()

        delta_tl = (self.scope.tl_corner - c)
        tl = c + max(delta_tl.length() - 0.01, eps) * delta_tl.normalize()

        delta_br = (self.scope.br_corner - c)
        br = c + max(delta_br.length() - 0.01, eps) * delta_br.normalize()

        p = Point(p.x / self.img_rect.width(), p.y / self.img_rect.height(), dtype=float)

        self.scope = translate_rect(Rectangle(tl.x, tl.y, br.x, br.y, dtype=float), (p - c) / 10)

    def translate_scope(self, p1, p2):
        p1 = Point(p1.x / self.img_rect.width(), p1.y / self.img_rect.height(), dtype=float)
        p2 = Point(p2.x / self.img_rect.width(), p2.y / self.img_rect.height(), dtype=float)
        self.scope = translate_rect(self.scope, p1 - p2)

    def add_transient_patch(self, patch, ax=None):
        self.transient_patches.append(patch)
        ax = ax or self.ax
        ax.add_patch(patch)
        self.refresh()

    def clear_transient_patch(self):
        for i in range(len(self.transient_patches)):
            self.transient_patches[i].remove()

        self.transient_patches.clear()
        self.refresh()

    def add_patch(self, patch, ax=None):
        self.patches.append(patch)
        ax = ax or self.ax
        ax.add_patch(patch)
        self.refresh()

    def hide_patches(self):
        for patch in self.patches:
            patch.set_visible(False)

    def show_patches(self):
        for patch in self.patches:
            patch.set_visible(True)

    def clear_patches(self):
        for i in range(len(self.patches)):
            self.patches[i].remove()

        self.patches.clear()
        self.refresh()

    def close(self):
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.root.quit()
        self.root.destroy()
        plt.close(self.fig)

    def get_axes_coordinates(self, event, dtype=int):
        if event.xdata is not None and event.ydata is not None:
            return Point(event.xdata, event.ydata, dtype=dtype)
        else:
            return None

    def mainloop(self):
        self.root.mainloop()
        return 0

    # named colors in matplotlib: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    # tk.TclError is raised when canvas.draw() is called while canvas is already closed by the user
    def on_enter_figure(self, event):
        try:
            if self.verbose:
                print('enter_figure', event.canvas.figure)
            event.canvas.figure.patch.set_facecolor('white')
            event.canvas.draw()
        except tk.TclError:
            pass

    def on_leave_figure(self, event):
        try:
            if self.verbose:
                print('enter_figure', event.canvas.figure)
            event.canvas.figure.patch.set_facecolor('whitesmoke')
            event.canvas.draw()
        except tk.TclError:
            pass

    def on_enter_axes(self, event):
        try:
            if self.verbose:
                print('enter_axes', event.inaxes)
            event.inaxes.patch.set_facecolor('lightyellow')
            event.canvas.draw()
        except tk.TclError:
            pass

    def on_leave_axes(self, event):
        try:
            if self.verbose:
                print('enter_axes', event.inaxes)
            event.inaxes.patch.set_facecolor('lightgoldenrodyellow')
            event.canvas.draw()
        except tk.TclError:
            pass

    def enable_callbacks(self):
        if not self.callbacks_alive:
            self.cids.append(
                self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('figure_enter_event', self.on_enter_figure)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('figure_leave_event', self.on_leave_figure)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            )
            self.cids.append(
                self.fig.canvas.mpl_connect('axes_leave_event', self.on_leave_axes)
            )

    def disable_callbacks(self):
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.cids.clear()

    def iconify(self):
        self.root.withdraw()

    def deiconify(self):
        self.root.deiconify()

    @property
    def callbacks_alive(self):
        return len(self.cids) > 0

    @property
    def window_center(self):
        return Point(
            self.root.winfo_x() + self.root.winfo_width() // 2,
            self.root.winfo_y() + self.root.winfo_height() // 2
        )

    def ask_multiple_choice_question(self, question: str, options: tuple):
        self.disable_callbacks()
        asker = MultipleChoiceQuestionAsker(
            question, options,
            *self.window_center
        )
        answer = asker.mainloop()
        self.enable_callbacks()

        return answer

    def ask_yes_no_question(self, question: str):
        self.disable_callbacks()
        asker = YesNoQuestionAsker(question, *self.window_center)
        answer = asker.mainloop()
        self.enable_callbacks()

        return answer

    def show_message(self, msg, title):
        self.disable_callbacks()
        notifier = MessageBox(msg, *self.window_center, title)
        notifier.mainloop()
        self.enable_callbacks()

    @on_caps_lock_off
    def on_key_press(self, event):
        if event.key == 'escape':
            self.close()
        elif event.key == 'f1':
            self.show_message(self.__class__.documentation(), 'Help')
        elif event.key == 'f5':
            self.scope = Rectangle(0.0, 0.0, 1.0, 1.0, dtype=float)
            self.adjust_view()
        if self.verbose:
            print('press', event.key)

    def on_key_release(self, event):
        if caps_lock_status():
            self.show_message('Caps Lock is turned on. Please turn it off.', 'Warning')
        if self.verbose:
            print('release', event.key)

    def on_mouse_press(self, event):
        p = self.get_axes_coordinates(event, float)
        if event.key == 'control' and event.button == 1 and event.inaxes is self.ax:
            self.panning_iptr.start_dragging(p)
        elif event.key == 'control' and event.button == 3 and event.inaxes is self.ax:
            self.zoom_iptr.start_dragging(p)

        if self.verbose:
            print(
                '%s click: button=%d, x=%d, y=%d, xdata={}, ydata={}, key=%s'
                .format(event.xdata, event.ydata) %
                (
                    'double' if event.dblclick else 'single',
                    event.button, event.x, event.y, event.key
                )
            )

    def on_mouse_move(self, event):
        if self.verbose:
            print(
                'mouse move: x=%d, y=%d, xdata={}, ydata={}, key=%s'
                .format(event.xdata, event.ydata) %
                (event.x, event.y, event.key)
            )
        self.clear_transient_patch()
        if self.panning_iptr.on_dragging:
            p = self.get_axes_coordinates(event, float)
            self.panning_iptr.update(p)
            self.translate_scope(self.panning_iptr.p1, self.panning_iptr.p2)
            self.adjust_view()
        elif self.zoom_iptr.on_dragging:
            p = self.get_axes_coordinates(event, float)
            self.zoom_iptr.update(p)
            self.add_transient_patch(
                self.zoom_iptr.rect.to_patch(
                    linewidth=1,
                    linestyle='--',
                    edgecolor='k',
                    facecolor='none'
                )
            )

    def on_mouse_release(self, event):
        if self.verbose:
            print(
                '%s release: button=%d, x=%d, y=%d, xdata={}, ydata={}, key=%s'
                .format(event.xdata, event.ydata) %
                (
                    'double' if event.dblclick else 'single',
                    event.button, event.x, event.y, event.key
                )
            )
        p = self.get_axes_coordinates(event, float)
        if self.panning_iptr.on_dragging:
            self.panning_iptr.finish_dragging(p)
            self.translate_scope(self.panning_iptr.p1, self.panning_iptr.p2)
            self.adjust_view()
        elif self.zoom_iptr.on_dragging:
            self.zoom_iptr.finish_dragging(p)
            self.clear_transient_patch()
            self.scope = self.get_scope(self.zoom_iptr.rect)
            self.adjust_view()

    @on_caps_lock_off
    def on_scroll(self, event):
        if self.verbose:
            print(
                'scroll: step=%s, key=%s' %
                (
                    'up' if event.step == 1 else 'down',
                    event.key
                )
            )
        p = self.get_axes_coordinates(event, float)
        if event.inaxes is self.ax and event.key == 'control':
            if event.step == 1:
                self.shrink_scope(p)
                self.adjust_view()
            else:
                self.grow_scope(p)
                self.adjust_view()
