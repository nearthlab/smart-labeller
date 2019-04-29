import json
import os

import cv2
import numpy as np

from .drag_interpreter import DragInterpreter
from .image_window import ImageWindow
from .image_group_viewer import ImageGroupViewer
from .mask_editor import MaskEditor
from .partially_labelled_dataset import PartiallyLabelledDataset, ObjectAnnotation, create_rgb_mask
from .utils import (random_colors, grabcut, fill_holes,
                    hide_axes_labels, on_caps_lock_off)


class LabelHelper(ImageGroupViewer):
    '''
<Object Actions>
w or up arrow: select the next object
s or down arrow: select the previous object
Ctrl + e: edit the current object mask
Ctrl + d: delete the current object
mouse right + dragging: add a new object
    '''

    def __init__(self, dataset: PartiallyLabelledDataset, info=None):
        assert dataset.root is not None, 'The dataset passed to LabelHelper is not loaded properly'
        super().__init__(
            [os.path.basename(image_file) for image_file in dataset.image_files],
            dataset.root,
            (0.05, 0.125, 0.75, 0.75)
        )
        self.dataset = dataset
        self.info = info or {}
        info_panel_top = 0.64
        if len(self.info) > 0:
            num_items = len(list(self.info.values())[0])
            info_panel_top = 0.14 + min(0.07 * num_items, 0.5)
            self.info_panel = self.fig.add_axes((0.81, 0.125, 0.14, min(0.07 * num_items, 0.5)))
            self.info_panel.set_facecolor('lightgoldenrodyellow')
            hide_axes_labels(self.info_panel)

        self.rgb_mask_panel = self.fig.add_axes((0.81, info_panel_top + 0.01, 0.125, 0.875 - info_panel_top))
        self.rgb_mask = np.array(())
        hide_axes_labels(self.rgb_mask_panel)

        self.mode = cv2.GC_INIT_WITH_RECT
        self.obj_pallete = random_colors(100)
        self.cls_pallete = random_colors(self.dataset.num_classes, bright=False, seed=6, uint8=True)
        self.rect_ipr = DragInterpreter()
        self.auto_grabcut = True
        self.obj_id = 0

        self.display()

    def set_items(self):
        return self.dataset.image_files

    def display(self):
        super().display()
        self.clear_patches()
        filename = os.path.basename(self.dataset.image_files[self.id])
        if self.should_update():
            self.img = self.dataset.load_image(self.id)
            self.annotations = self.dataset.load_annotations(self.id)
            self.set_image(self.img)
            self.obj_id = 0

            if filename in self.info:
                self.info_panel.clear()

                def resolve_lines(s, max_len, max_lines=6):
                    q, r = len(s).__divmod__(max_len)
                    lines = [s[i * max_len:(i + 1) * max_len] for i in range(q)]
                    if r > 0:
                        lines.append(s[-r:])
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        lines[-1] = lines[-1][-3] + '...'
                    return '\n'.join(lines)

                keys = list(sorted(self.info[filename].keys()))
                values = [self.info[filename][key] for key in keys]
                for i, items in enumerate(zip(keys, values)):
                    self.info_panel.text(
                        0.02, 0.9 - i * 0.2,
                        resolve_lines(items[0], 14),
                        bbox=dict(
                            linewidth=1, alpha=0.0,
                            edgecolor='none',
                            facecolor='none',
                        )
                    )
                    self.info_panel.text(
                        0.48, 0.93 - i * 0.2,
                        resolve_lines(items[1], 20),
                        bbox=dict(
                            linewidth=1, alpha=0.0,
                            edgecolor='none',
                            facecolor='none',
                        ),
                        verticalalignment='top'
                    )

        for obj_id, annotation in enumerate(self.annotations):
            alpha = 0.3 if obj_id == self.obj_id else 0.1
            linewidth = 3 if obj_id == self.obj_id else 1
            color = self.obj_pallete[obj_id]
            bbox = annotation.bbox
            self.add_patch(bbox.to_patch(
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            ))
            for poly in annotation.polys:
                self.add_patch(
                    poly.to_patch(
                        self.img.shape[:2],
                        linewidth=linewidth,
                        edgecolor=color,
                        facecolor=(*color, alpha),
                    ))
            self.patches.append(self.ax.text(
                *bbox.tl_corner,
                '{}. {}'.format(obj_id, self.dataset.class_id2name[self.annotations[obj_id].class_id]),
                bbox=dict(facecolor=color, alpha=alpha)
            ))

        self.rgb_mask = create_rgb_mask(self.annotations, self.cls_pallete, self.img.shape)
        self.rgb_mask_panel.clear()
        self.rgb_mask_panel.imshow(self.rgb_mask)

        self.ax.set_title(
            'FILENAME: {} | IMAGE ID: {} | OBJECT ID: {} | OBJECT CLASS: {}'.format(
                filename, self.id, self.obj_id,
                self.dataset.class_id2name[self.annotations[self.obj_id].class_id]
            ) if len(self.annotations) > self.obj_id else 'FILENAME: {} | IMAGE ID: {} | NO OBJECT'.format(
                filename, self.id
            )
        )

    def on_image_menubar_select(self, event):
        super().on_image_menubar_select(event)
        self.save_current_labels()

    def save_current_labels(self):
        prev_annotations = self.dataset.load_annotations(self.id)
        if prev_annotations != self.annotations:
            label_path = self.dataset.infer_label_path(self.id)
            if len(self.annotations) == 0:
                os.remove(label_path)
            else:
                with open(label_path, 'w') as fp:
                    json.dump([annotation.json() for annotation in self.annotations], fp)

    def remove_current_object(self):
        if self.obj_id < len(self.annotations):
            del self.annotations[self.obj_id]
            self.obj_id = max(0, len(self.annotations) - 1)

    def ask_class_id(self):
        return self.ask_multiple_choice_question('Which class does this object belong to?', tuple(self.dataset.class_id2name))

    def mask_editor_session(self):
        self.disable_callbacks()

        if self.obj_id < len(self.annotations):
            self.disable_menubar()
            self.iconify()
            mask_touch_helper = MaskEditor(self.img, self.annotations[self.obj_id].mask(self.img.shape[:2]), win_title=os.path.basename(self.dataset.image_files[self.id]))
            mask = mask_touch_helper.mainloop()
            self.deiconify()
            self.enable_menubar()

            if mask is not None:
                answer = self.ask_multiple_choice_question('Save edited mask as:', ('Overwrite the current object mask', 'Add as a new object', 'Add each component as a new object', 'Do not save'))
                if answer == 0:
                    self.annotations[self.obj_id] = ObjectAnnotation(
                        np.where(mask % 2 == 1, 255, 0).astype('uint8'),
                        self.annotations[self.obj_id].class_id
                    )
                elif answer == 1:
                    class_id = self.ask_class_id()
                    if class_id != -1:
                        self.annotations.insert(
                            self.obj_id + 1,
                            ObjectAnnotation(
                                np.where(mask % 2 == 1, 255, 0).astype(np.uint8),
                                class_id
                            )
                        )
                        self.obj_id += 1
                elif answer == 2:
                    class_id = self.ask_class_id()
                    if class_id != -1:
                        from .utils import ConnectedComponents
                        comps = ConnectedComponents(np.where(mask % 2 == 1, 255, 0).astype(np.uint8))
                        for i in range(len(comps)):
                            self.annotations.insert(
                                self.obj_id + i + 1,
                                ObjectAnnotation(
                                    comps.mask(i),
                                    class_id
                                )
                            )
        else:
            self.show_message('Please add an object by drawing a rectangle first', 'Guide')

        self.enable_callbacks()
        self.force_focus()

    @on_caps_lock_off
    def on_key_press(self, event):
        if event.key in ['left', 'right', 'a', 'd', 'escape']:
            self.save_current_labels()
            self.obj_id = 0
            super().on_key_press(event)
            if event.key != 'escape':
                self.display()
        else:
            super().on_key_press(event)
            if event.key in ['w', 'up']:
                if len(self.annotations) > 0:
                    self.obj_id = (self.obj_id + 1) % len(self.annotations)
            elif event.key in ['s', 'down']:
                if len(self.annotations) > 0:
                    self.obj_id = (self.obj_id - 1) % len(self.annotations)
            elif event.key in ['W', 'shift+up']:
                if len(self.annotations) > 1:
                    next_obj_id = (self.obj_id + 1) % len(self.annotations)
                    self.annotations[self.obj_id], self.annotations[next_obj_id] = self.annotations[next_obj_id], self.annotations[self.obj_id]
                    self.obj_id = next_obj_id
            elif event.key in ['S', 'shift+down']:
                if len(self.annotations) > 1:
                    prev_obj_id = (self.obj_id + 1) % len(self.annotations)
                    self.annotations[self.obj_id], self.annotations[prev_obj_id] = self.annotations[prev_obj_id], self.annotations[self.obj_id]
                    self.obj_id = prev_obj_id
            elif event.key == 'ctrl+d':
                self.remove_current_object()
            elif event.key == 'ctrl+e':
                self.mask_editor_session()
            elif event.key == 'ctrl+a':
                class_id = self.ask_class_id()
                if class_id != -1:
                    self.annotations.append(ObjectAnnotation(
                        self.img_rect.to_mask(self.img.shape[:2]),
                        class_id
                    ))
            elif event.key == 'ctrl+D':
                answer = self.ask_yes_no_question('Do you want to delete the current image file?')
                if answer:
                    os.remove(self.dataset.image_files[self.id])
                    label_path = self.dataset.infer_label_path(self.id)
                    if os.path.isfile(label_path):
                        os.remove(label_path)
                    self.dataset.load(self.dataset.root)
                    self.prev_id = (self.id - 1) % self.num_items
            elif event.key == 'm':
                class_id = self.ask_class_id()
                if class_id != -1:
                    self.annotations[self.obj_id].class_id = class_id
            elif event.key == 'j':
                first_unlabeled = 0
                for i in range(self.num_items):
                    if os.path.isfile(self.dataset.infer_label_path(i)):
                        continue
                    else:
                        first_unlabeled = i
                        break
                self.prev_id, self.id = self.id, (first_unlabeled - 1) % self.num_items

            self.display()

    def on_mouse_press(self, event):
        super().on_mouse_press(event)
        p = self.get_axes_coordinates(event)
        if event.button == 3 and event.key != 'control' and event.inaxes is self.ax:
            self.rect_ipr.start_dragging(p)
            self.auto_grabcut = (event.key != 'shift')

    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        p = self.get_axes_coordinates(event)
        if self.rect_ipr.on_dragging:
            self.rect_ipr.update(p)
            self.add_transient_patch(self.rect_ipr.rect.to_patch(
                linewidth=1,
                linestyle='--',
                edgecolor='b',
                facecolor='none'
            ))
        elif event.inaxes is self.rgb_mask_panel and p in self.img_rect:
            match = np.where((self.cls_pallete == self.rgb_mask[p.y][p.x]).all(axis=-1))[0]
            class_name = 'Background' if len(match) == 0 else self.dataset.class_id2name[match[0]]

            self.transient_patches.append(
                self.rgb_mask_panel.text(
                    0, -1,
                    class_name,
                    bbox=dict(
                        linewidth=1, alpha=0.0,
                        edgecolor='none',
                        facecolor='none',
                    )
                )
            )

    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        p = self.get_axes_coordinates(event)
        if self.rect_ipr.on_dragging:
            self.rect_ipr.finish_dragging(p)

            self.clear_transient_patch()
            rect = self.rect_ipr.rect.intersect(self.img_rect)
            if rect.tl_corner != rect.br_corner:
                class_id = self.ask_class_id()
                if class_id != -1:
                    rect_mask = rect.to_mask(self.img.shape[:2])

                    if self.auto_grabcut:
                        mask = grabcut(self.img, cv2.GC_INIT_WITH_RECT, rect=self.rect_ipr.rect)
                        # If the initial grabcut fails to find any foreground pixels
                        # set the mask as the rectangle region itself
                        if np.array_equal(mask % 2 == 1, np.zeros_like(mask)):
                            self.annotations.append(ObjectAnnotation(
                                rect_mask, class_id
                            ))
                        else:
                            self.annotations.append(ObjectAnnotation(
                                np.where(mask % 2 == 1, 255, 0).astype('uint8'),
                                class_id
                            ))
                    else:
                        self.annotations.append(ObjectAnnotation(
                            rect_mask, class_id
                        ))

                    self.obj_id = len(self.annotations) - 1
                    self.display()
