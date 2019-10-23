import json
import os

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

        # defaul to hard-coded value if the radius becomes too large
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
    asdf
    가나다라
    '''
    INACTIVE_AXES_COLOR = 'lightyellow'
    INTERMED_AXES_COLOR = 'greenyellow'
    ACTIVE_AXES_COLOR = 'limegreen'

    INACTIVE_COLOR = 'darkgray'
    ACTIVE_COLOR = 'limegreen'
    INACTIVE_BUTTON_COLOR = 'gainsboro'
    ACTIVE_BUTTON_COLOR = 'royalblue'

    NOT_SPECIFIED_VALUE = ''

    def __init__(self, cat_path):
        with open(cat_path, 'r') as fp:
            self.categories = json.load(fp)

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
            axes_pos=(0.05, 0.05, 0.6, 0.9)
        )
        hide_axes_labels(self.ax)
        self.ax.set_facecolor(TagHelper.INACTIVE_AXES_COLOR)
        self.set_title(os.path.basename(self.root_dir))

        verify_or_create_directory(self.tags_dir)

        cat_names = sorted(self.categories.keys(), reverse=True)
        num_cats = len(self.categories)
        num_options = [len(self.categories.get(category)) for category in cat_names]

        x = 0.7
        width = 0.25
        dy = 1 / (sum(num_options) + len(num_options) + num_cats + 1)
        self.radius = dy / 2

        self.focused_dialog_id = 0
        self.dialogs = dict()
        # create panels from the bottom to the top
        for idx, category in enumerate(cat_names):
            y = (sum(num_options[:idx]) + 2 * idx + 1) * dy
            height = (num_options[idx] + 1) * dy
            panel, buttons = self.create_panel(
                category, self.categories.get(category),
                (x, y, width, height),
                callback=lambda x: self.syncronize_axes()
            )
            self.dialogs[category] = {
                'panel': panel,
                'buttons': buttons
            }

        # category of the panels from the top to the bottom
        self.ordered_categories = list(reversed(cat_names))

        self.clipboard = dict()
        self.history = dict()
        self.display()

    def create_panel(self, category, options, pos, callback=None):
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
        buttons.on_clicked(callback or (lambda x: self.syncronize_axes()))
        return panel, buttons

    @property
    def default_annotation(self):
        return {
            'file_name': self.image_name,
            'tag': {
                category: TagHelper.NOT_SPECIFIED_VALUE
                for category in self.categories
            }
        }

    @property
    def image_name(self):
        return os.path.split(self.items[self.id])[1]

    @property
    def tag_name(self):
        name, ext = os.path.splitext(self.image_name)
        return '{}.json'.format(name)

    def is_completed(self, id):
        image_name = os.path.split(self.items[id])[1]
        name, ext = os.path.splitext(image_name)
        tag_name = '{}.json'.format(name)
        tag_path = os.path.join(self.tags_dir, tag_name)

        try:
            with open(tag_path, 'r') as fp:
                annotation = json.load(fp)
                tag = annotation.get('tag')
                for value in tag.values():
                    if value == '':
                        return False
                return True
        except:
            return False

    def syncronize_axes(self):
        isSpecified = []

        for category in self.dialogs:
            dialog = self.dialogs.get(category)
            buttons = dialog.get('buttons')
            value = buttons.value_selected

            try:
                index = self.categories.get(category).index(value) + 1
            except:
                index = 0  # not specified

            isSpecified.append(index > 0)

            panel = dialog.get('panel')
            panel.set_facecolor(
                TagHelper.INACTIVE_COLOR if index == 0
                else TagHelper.ACTIVE_COLOR
            )

        numSpecified = sum(isSpecified)
        if numSpecified == 0:
            self.ax.set_facecolor(TagHelper.INACTIVE_AXES_COLOR)
        elif numSpecified < len(self.categories):
            self.ax.set_facecolor(TagHelper.INTERMED_AXES_COLOR)
        else:
            self.ax.set_facecolor(TagHelper.ACTIVE_AXES_COLOR)

        for idx, category in enumerate(self.ordered_categories):
            color = 'crimson' if self.focused_dialog_id == idx else 'none'
            set_border_color(self.dialogs.get(category).get('panel'), color)

        self.refresh()

    @property
    def annotation(self):
        '''
        :return: annotation represented by the buttons
        '''
        return {
            'file_name': self.image_name,
            'tag': {
                category: dialog.get('buttons').value_selected
                for category, dialog in self.dialogs.items()
            }
        }

    @annotation.setter
    def annotation(self, annotation):
        '''
        Sets the buttons by the given annotation
        :param annotation: dict
        :return: None
        '''
        for category in self.dialogs:
            value = TagHelper.NOT_SPECIFIED_VALUE
            tag = annotation.get('tag')
            if tag is not None:
                value = tag.get(category) or value

            try:
                index = self.categories.get(category).index(value) + 1
            except:
                index = 0  # not specified

            dialog = self.dialogs.get(category)
            buttons = dialog.get('buttons')
            buttons.set_active(index)
        self.syncronize_axes()

    @property
    def saved_annotation(self):
        '''
        :return: annotation written on the disk if annotation file is found, otherwise default annotation
        '''
        try:
            with open(os.path.join(self.tags_dir, self.tag_name), 'r') as fp:
                ann = json.load(fp)
        except:
            ann = self.default_annotation
        return ann

    @saved_annotation.setter
    def saved_annotation(self, annotation):
        '''
        Saves the given annotation to the disk
        if it is different from default annotation and the annotation currently written in the disk
        :param annotation: dict
        :return: None
        '''
        tag_path = os.path.join(self.tags_dir, self.tag_name)
        if annotation != self.saved_annotation:
            if annotation == self.default_annotation and os.path.isfile(tag_path):
                if self.ask_yes_no_question('이 이미지({})에 대한 레이블을 삭제하시겠습니까?'.format(self.image_name)):
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

    @on_caps_lock_off
    def on_key_press(self, event):
        if event.key in ['left', 'right', 'a', 'd', 'escape', 'home', 'end']:
            self.saved_annotation = self.annotation
            super().on_key_press(event)
            if event.key != 'escape':
                self.display()
                self.focused_dialog_id = 0
                self.syncronize_axes()
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
                    if not self.is_completed(id):
                        if id == self.id:
                            self.show_message('이미 완료되지 않은 가장 첫 번째 이미지를 보고 있습니다.', 'Info')
                            return
                        else:
                            self.id = id
                            self.display()
                            return
                self.show_message('모든 이미지의 레이블링이 완료된 상태입니다!', 'Info')
            elif event.key in ['up', 'down', 's', 'w']:
                annotation = self.annotation
                category = self.ordered_categories[self.focused_dialog_id]
                options = [TagHelper.NOT_SPECIFIED_VALUE] + self.categories.get(category)
                value = annotation.get('tag').get(category)
                option_id = options.index(value)
                option_id = (option_id + 1) % len(options) if event.key in ['down', 's'] else (option_id - 1) % len(options)
                annotation['tag'][category] = options[option_id]
                self.annotation = annotation
            elif event.key == 'enter':
                self.focused_dialog_id = (self.focused_dialog_id + 1) % len(self.dialogs)
                self.syncronize_axes()
            elif event.key == 'ctrl+enter':
                self.focused_dialog_id = (self.focused_dialog_id - 1) % len(self.dialogs)
                self.syncronize_axes()
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
                    self.focused_dialog_id = 0
                    self.syncronize_axes()
