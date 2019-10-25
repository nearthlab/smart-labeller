import json
import os
from functools import partial

import numpy as np
from matplotlib.widgets import (
    AxesWidget, Circle, RadioButtons
)

from ..base import (
    ImageGroupViewer, on_caps_lock_off,
    load_rgb_image, hide_axes_labels,
    verify_or_create_directory
)


class CustomRadioButtons(RadioButtons):
    """
        A GUI neutral radio button.

        For the buttons to remain responsive
        you must keep a reference to this object.

        The following attributes are exposed:

         *ax*
            The :class:`matplotlib.axes.Axes` instance the buttons are in

         *activecolor*
            The color of the button when clicked

         *labels*
            A list of :class:`matplotlib.text.Text` instances

         *circles*
            A list of :class:`matplotlib.patches.Circle` instances

         *value_selected*
            A string listing the current value selected

        Connect to the RadioButtons with the :meth:`on_clicked` method
        """

    def __init__(self, ax, labels, active=0, activecolor='blue', inactivecolor=None):
        """
        Add radio buttons to :class:`matplotlib.axes.Axes` instance *ax*

        *labels*
            A len(buttons) list of labels as strings

        *active*
            The index into labels for the button that is active

        *activecolor*
            The color of the button when clicked
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        dy = 1. / (len(labels) + 1)
        ys = np.linspace(1 - dy, dy, len(labels))
        cnt = 0
        axcolor = ax.get_facecolor()
        self.inactivecolor = inactivecolor or axcolor

        # scale the radius of the circle with the spacing between each one
        circle_radius = (dy / 2) - 0.01

        # default to hard-coded value if the radius becomes too large
        if (circle_radius > 0.05):
            circle_radius = 0.05

        self.labels = []
        self.circles = []
        for y, label in zip(ys, labels):
            t = ax.text(0.25, y, label, transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='center')

            if cnt == active:
                self.value_selected = label
                facecolor = self.activecolor
            else:
                facecolor = self.inactivecolor

            p = Circle(xy=(0.15, y), radius=circle_radius, edgecolor='black',
                       facecolor=facecolor, transform=ax.transAxes)

            self.labels.append(t)
            self.circles.append(p)
            ax.add_patch(p)
            cnt += 1

        self.connect_event('button_press_event', self._clicked)

        self.cnt = 0
        self.observers = {}

    def set_active(self, index):
        """
        Trigger which radio button to make active.

        *index* is an index into the original label list
            that this object was constructed with.
            Raise ValueError if the index is invalid.

        Callbacks will be triggered if :attr:`eventson` is True.

        """
        if 0 > index >= len(self.labels):
            raise ValueError("Invalid RadioButton index: %d" % index)

        self.value_selected = self.labels[index].get_text()

        for i, p in enumerate(self.circles):
            if i == index:
                color = self.activecolor
            else:
                color = self.inactivecolor
            p.set_facecolor(color)

        if self.drawon:
            self.ax.figure.canvas.draw()

        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(self.labels[index].get_text())


def set_border_color(ax, color):
    for pos in ['left', 'top', 'right', 'bottom']:
        ax.spines[pos].set_color(color)


class TagHelper(ImageGroupViewer):
    '''
    [이미지 이동]
    오른쪽 이미지 리스트에서 마우스로 파일 이름을 직접 선택하거나
    다음 단축키들을 이용하여 이동할 수 있습니다.
    방향키 →, d: 다음 이미지로 넘기기
    방향키 ←, a: 이전 이미지로 넘기기
    Home: 첫 번째 이미지로 이동
    End: 마지막 이미지로 이동
    Ctrl + F: 레이블이 완료되지 않은 가장 첫 번째 이미지를 찾아서 이동

    [박스 이동]
    마우스로 직접 선택하여 원하는 박스를 선택하거나
    마우스 휠을 이용하여 위/아래 박스로 이동할 수 있습니다.
    혹은 다음 단축키들을 활용할 수 있습니다.
    Pgup: 아래쪽 박스로 이동
    Pgdn: 위쪽 박스로 이동

    [레이블 변경]
    각각의 박스 안에 있는 레이블을 직접 클릭하거나
    다음 단축키들을 이용하여 레이블을 변경시킬 수 있습니다.
    방향키 ↑, w: 위쪽 옵션으로 이동
    방향키 ↓, s: 아래쪽 옵션으로 이동
    BackSpace, Delete: 현재 레이블을 모두 초기화

    [그 외 기능]
    Esc: 저장 후 종료
    Ctrl + C: 현재 레이블을 복사하기
    Ctrl + V: 복사해둔 레이블 붙여넣기
    Ctrl + BackSpace, Ctrl + Delete: 현재 이미지를 제외하기
    '''
    INACTIVE_AXES_COLOR = 'lightyellow'
    INTERMED_AXES_COLOR = 'greenyellow'
    ACTIVE_AXES_COLOR = 'limegreen'
    INACTIVE_TKINTER_COLOR = 'light yellow'
    INTERMED_TKINTER_COLOR = 'green yellow'
    ACTIVE_TKINTER_COLOR = 'lime green'

    INACTIVE_COLOR = 'darkgray'
    ACTIVE_COLOR = 'limegreen'
    INACTIVE_BUTTON_COLOR = 'gainsboro'
    ACTIVE_BUTTON_COLOR = 'royalblue'

    NOT_SPECIFIED_VALUE = ''

    @classmethod
    def num_specified(cls, annotation):
        num = 0
        tag = annotation.get('tag')
        for value in tag.values():
            if value != cls.NOT_SPECIFIED_VALUE:
                num += 1
        return num

    def __init__(self, cat_path):
        with open(cat_path, 'r') as fp:
            self.__json = json.load(fp)

        self.root_dir, cat_file = os.path.split(cat_path)
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.tags_dir = os.path.join(self.root_dir, 'tags')
        if not os.path.isdir(self.image_dir):
            raise Exception(
                '선택한 {} 파일과 같은 폴더 안에 레이블 하고자 하는 이미지들을 담은 "images"라는 폴더가 있어야 합니다.'.
                    format(cat_file)
            )

        super(TagHelper, self).__init__(
            sorted([os.path.join(self.image_dir, x)
                    for x in os.listdir(self.image_dir)
                    if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')
                    ]),
            os.path.basename(cat_path),
            axes_pos=(0.05, 0.05, 0.6, 0.9),
            menubar_kwargs={'selectbackground': 'light sky blue'}
        )

        self.image_menubar.listbox.pack()
        hide_axes_labels(self.ax)
        self.ax.set_facecolor(TagHelper.INACTIVE_AXES_COLOR)
        self.set_title(os.path.basename(self.root_dir))

        verify_or_create_directory(self.tags_dir)

        cat_names = sorted(self.__json.keys(), reverse=True)
        num_cats = len(self.__json)
        num_options = [len(self.__json.get(category)) for category in cat_names]

        x = 0.7
        width = 0.25
        dy = 1 / (sum(num_options) + len(num_options) + num_cats + 1)
        ys = [(sum(num_options[:idx]) + 2 * idx + 1) * dy for idx in range(len(cat_names))]
        heights = [(num_options[idx] + 1) * dy for idx in range(len(cat_names))]
        self.radius = dy / 2

        # sort assets from the top to the bottom
        self.names = list(reversed(cat_names))
        ys = list(reversed(ys))
        heights = list(reversed(heights))
        self.options = [[TagHelper.NOT_SPECIFIED_VALUE] + self.__json.get(name) for name in self.names]

        self.panels = []
        self.buttons = []
        for (idx, name), y, height, options in zip(enumerate(self.names), ys, heights, self.options):
            panel = self.fig.add_axes((x, y, width, height))
            for pos in ['left', 'top', 'right', 'bottom']:
                panel.spines[pos].set_linewidth(2)
            panel.set_title(name)
            panel.set_facecolor(TagHelper.INACTIVE_COLOR)
            hide_axes_labels(panel)

            button = CustomRadioButtons(
                panel, options,
                activecolor=TagHelper.ACTIVE_BUTTON_COLOR,
                inactivecolor=TagHelper.INACTIVE_BUTTON_COLOR
            )
            for circle in button.circles:
                circle.update({'radius': self.radius})
            button.on_clicked(partial(self.sync_panel, idx=idx))
            self.panels.append(panel)
            self.buttons.append(button)

        self.clipboard = dict()
        self.focused_panel_idx = 0

        for idx, panel in enumerate(self.panels):
            set_border_color(panel, 'crimson' if idx == 0 else 'none')

        self.load_progress()
        self.display()

    @property
    def num_categories(self):
        return len(self.names)

    def create_dialog_box(self, category, options, pos, callback):
        panel = self.fig.add_axes(pos)
        for pos in ['left', 'top', 'right', 'bottom']:
            panel.spines[pos].set_linewidth(2)
        panel.set_title(category)
        panel.set_facecolor(TagHelper.INACTIVE_COLOR)

        hide_axes_labels(panel)
        buttons = CustomRadioButtons(
            panel,
            [TagHelper.NOT_SPECIFIED_VALUE] + options,
            activecolor=TagHelper.ACTIVE_BUTTON_COLOR,
            inactivecolor=TagHelper.INACTIVE_BUTTON_COLOR
        )
        for circle in buttons.circles:
            circle.update({'radius': self.radius})
        buttons.on_clicked(callback)
        return panel, buttons

    def sync_panel(self, value, idx):
        panel = self.panels[idx]
        options = self.options[idx]
        option_idx = options.index(value)
        panel.set_facecolor(
            TagHelper.INACTIVE_COLOR if option_idx == 0
            else TagHelper.ACTIVE_COLOR
        )

    def load_progress(self):
        for id in range(self.num_items):
            self.sync_menubar_progress(id, self.load_annotation(id), False)
        self.image_menubar.listbox.pack()

    def sync_ax_progress(self):
        num_specified = TagHelper.num_specified(self.annotation)
        color = TagHelper.INACTIVE_AXES_COLOR if num_specified == 0 else \
            TagHelper.INTERMED_AXES_COLOR if num_specified < self.num_categories else \
                TagHelper.ACTIVE_AXES_COLOR
        self.ax.set_facecolor(color)
        self.refresh()

    def sync_menubar_progress(self, id, annotation, pack=True):
        num_specified = TagHelper.num_specified(annotation)
        color = TagHelper.INACTIVE_TKINTER_COLOR if num_specified == 0 else \
            TagHelper.INTERMED_TKINTER_COLOR if num_specified < self.num_categories else \
                TagHelper.ACTIVE_TKINTER_COLOR
        self.image_menubar.listbox.itemconfig(id, bg=color)
        if pack:
            self.image_menubar.listbox.pack()

    @property
    def default_annotation(self):
        return {
            'file_name': self.image_name,
            'tag': {
                name: TagHelper.NOT_SPECIFIED_VALUE
                for name in self.names
            }
        }

    @property
    def image_name(self):
        return self.get_image_name(self.id)

    def get_image_name(self, id):
        return os.path.split(self.items[id])[1]

    @property
    def tag_name(self):
        return self.get_tag_name(self.id)

    def get_tag_name(self, id):
        name, ext = os.path.splitext(self.get_image_name(id))
        return '{}.json'.format(name)

    def set_panel_focus(self, panel_idx):
        set_border_color(self.panels[self.focused_panel_idx], 'none')
        set_border_color(self.panels[panel_idx], 'crimson')
        self.focused_panel_idx = panel_idx
        self.refresh()

    def scroll_panel_focus(self, downward: bool):
        if downward:
            panel_idx = (self.focused_panel_idx + 1) % self.num_categories
        else:
            panel_idx = (self.focused_panel_idx - 1) % self.num_categories

        self.set_panel_focus(panel_idx)

    def scroll_option(self, downward: bool):
        annotation = self.annotation
        name = self.names[self.focused_panel_idx]
        options = self.options[self.focused_panel_idx]
        option_id = options.index(annotation.get('tag').get(name))
        option_id = (option_id + 1) % len(options) if downward else (option_id - 1) % len(options)
        annotation['tag'][name] = options[option_id]
        self.annotation = annotation

    @property
    def annotation(self):
        '''
        :return: annotation represented by the buttons
        '''
        return {
            'file_name': self.image_name,
            'tag': {
                name: button.value_selected
                for name, button in zip(self.names, self.buttons)
            }
        }

    @annotation.setter
    def annotation(self, annotation):
        '''
        Sets the buttons by the given annotation
        :param annotation: dict
        :return: None
        '''
        for i, name in enumerate(self.names):
            value = annotation.get('tag').get(name)
            self.buttons[i].set_active(self.options[i].index(value))
        self.sync_ax_progress()

    def load_annotation(self, id):
        '''
        :param id: item id
        :return: annotation with the given id written on the disk if annotation file is found,
                otherwise default annotation
        '''
        try:
            with open(os.path.join(self.tags_dir, self.get_tag_name(id)), 'r') as fp:
                ann = json.load(fp)
        except:
            ann = self.default_annotation
        return ann

    @property
    def saved_annotation(self):
        '''
        :return: annotation with the current id written on the disk if annotation file is found,
                otherwise default annotation
        '''
        return self.load_annotation(self.id)

    @saved_annotation.setter
    def saved_annotation(self, annotation):
        '''
        Saves the given annotation to the disk
        if it is different from default annotation and the annotation currently written in the disk
        :param annotation: dict
        :return: None
        '''
        self.sync_menubar_progress(self.id, self.annotation)
        tag_path = os.path.join(self.tags_dir, self.tag_name)
        if annotation != self.saved_annotation:
            if annotation == self.default_annotation and os.path.isfile(tag_path):
                os.remove(tag_path)
            else:
                with open(tag_path, 'w') as fp:
                    json.dump(annotation, fp)
        elif annotation == self.default_annotation and os.path.isfile(tag_path):
            os.remove(tag_path)

    '''
    Override parent class methods
    '''

    @classmethod
    def documentation(cls):
        return cls.__doc__

    def on_enter_figure(self, event):
        pass

    def on_leave_figure(self, event):
        pass

    def on_enter_axes(self, event):
        pass

    def on_leave_axes(self, event):
        pass

    def enable_callbacks(self):
        super(TagHelper, self).enable_callbacks()
        if hasattr(self, 'buttons'):
            for button in self.buttons:
                button.connect_event('button_press_event', button._clicked)

    def disable_callbacks(self):
        super(TagHelper, self).disable_callbacks()
        if hasattr(self, 'buttons'):
            for button in self.buttons:
                button.disconnect_events()

    def remove_current_item(self):
        super(TagHelper, self).remove_current_item()
        self.load_progress()

    def display(self):
        super(TagHelper, self).display()
        if self.should_update():
            image_file = self.items[self.id]
            img = load_rgb_image(image_file)
            self.set_image(img)

            title = "{} ({}/{})".format(
                os.path.basename(image_file),
                self.id + 1,
                self.num_items
            )
            self.ax.set_title(title)
            self.annotation = self.saved_annotation

    def on_image_menubar_select(self, event):
        self.saved_annotation = self.annotation
        super(TagHelper, self).on_image_menubar_select(event)

    def on_scroll(self, event):
        self.scroll_panel_focus(event.step == -1)

    def on_mouse_press(self, event):
        super(TagHelper, self).on_mouse_press(event)
        if event.inaxes in self.panels:
            self.set_panel_focus(self.panels.index(event.inaxes))

    @on_caps_lock_off
    def on_key_press(self, event):
        if event.key in ['left', 'right', 'a', 'd', 'escape', 'home', 'end']:
            self.saved_annotation = self.annotation
            super().on_key_press(event)
            if event.key != 'escape':
                self.display()
        else:
            super().on_key_press(event)
            if event.key in ['delete', 'backspace']:
                self.annotation = self.default_annotation
            elif event.key == 'ctrl+c':
                self.clipboard = self.annotation.get('tag')
            elif event.key == 'ctrl+v':
                annotation = self.annotation
                annotation.get('tag').update(self.clipboard)
                self.annotation = annotation
            elif event.key == 'ctrl+f':
                for id in range(self.num_items):
                    if TagHelper.num_specified(self.load_annotation(id)) < self.num_categories:
                        if id == self.id:
                            self.show_message('이미 완료되지 않은 가장 첫 번째 이미지를 보고 있습니다.', 'Info')
                            return
                        else:
                            self.saved_annotation = self.annotation
                            self.id = id
                            self.display()
                            return
                self.show_message('모든 이미지의 레이블링이 완료된 상태입니다!', 'Info')
            elif event.key in ['up', 'w', 'down', 's']:
                self.scroll_option(event.key in ['down', 's'])
            elif event.key in ['pageup', 'pagedown']:
                self.scroll_panel_focus(event.key == 'pagedown')
            elif event.key in ['ctrl+delete', 'ctrl+backspace']:
                if self.ask_yes_no_question('이 이미지를 제외하시겠습니까?'):
                    excluded_dir = os.path.join(self.root_dir, 'excluded')
                    verify_or_create_directory(excluded_dir)
                    os.rename(
                        os.path.join(self.image_dir, self.image_name),
                        os.path.join(excluded_dir, self.image_name)
                    )

                    tag_path = os.path.join(self.tags_dir, self.tag_name)
                    if os.path.isfile(tag_path):
                        os.remove(tag_path)

                    self.remove_current_item()
                    self.display()
