import os
import cv2
import json
import numpy as np

from .drag_interpreter import DragInterpreter
from .mask_modifier import MaskModifier
from .image_group_viewer import ImageGroupViewer
from .partially_labelled_dataset import PartiallyLabelledDataset, ObjectAnnotation
from .utils import random_colors, grabcut



class LabelHelper(ImageGroupViewer):
    '''
<Object Actions>
w or up arrow: select the next object
s or down arrow: select the previous object
Ctrl + e: edit the current object mask
Ctrl + d: delete the current object
Shift + mouse right + dragging: add a new object
    '''
    def __init__(self, dataset: PartiallyLabelledDataset):
        assert dataset.root is not None, 'The dataset passed to LabelHelper is not loaded properly'
        super().__init__(
            [os.path.basename(image_file) for image_file in dataset.image_files],
            dataset.name
        )
        self.dataset = dataset
        self.mode = cv2.GC_INIT_WITH_RECT
        self.pallete = random_colors(100)
        self.rect_ipr = DragInterpreter()
        self.obj_id = 0

        self.display()


    def set_items(self):
        return self.dataset.image_files


    def display(self):
        super().display()
        if self.should_update():
            self.img = self.dataset.load_image(self.id)
            self.annotations = self.dataset.load_annotations(self.id)
            self.set_image(self.img)

        self.clear_patches()
        for obj_id, annotation in enumerate(self.annotations):
            alpha = 0.7 if obj_id == self.obj_id else 0.3
            linewidth = 2 if obj_id == self.obj_id else 1
            color = self.pallete[obj_id]
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

        
        title = 'FILENAME: {} | IMAGE ID: {} | OBJECT ID: {} | OBJECT CLASS: {}'.format(
            os.path.basename(self.dataset.image_files[self.id]),
            self.id,
            self.obj_id,
            self.dataset.class_id2name[self.annotations[self.obj_id].class_id]
        ) if len(self.annotations) > 0 else 'FILENAME: {} | IMAGE ID: {} | NO OBJECT'.format(
            os.path.basename(self.dataset.image_files[self.id]),
            self.id
        )
        self.set_title(title)


    def on_image_menubar_select(self, event):
        super().on_image_menubar_select(event)
        self.save_current_labels()


    def save_current_labels(self):
        with open(self.dataset.infer_label_path(self.id), 'w') as fp:
            json.dump([annotation.json(self.img.shape[:2]) for annotation in self.annotations], fp)


    def remove_current_object(self):
        if self.obj_id < len(self.annotations):
            del self.annotations[self.obj_id]
            self.obj_id = max(0, len(self.annotations) - 1)


    def ask_class_id(self):
        return self.ask_multiple_choice_question('Which class does this object belong to?', tuple(self.dataset.class_id2name))


    def mask_touch_session(self):
        self.disable_callbacks()

        if self.obj_id < len(self.annotations):
            self.disable_menubar()
            self.iconify()
            mask_touch_helper = MaskModifier(self.img, self.annotations[self.obj_id].mask(self.img.shape[:2]))
            mask = mask_touch_helper.mainloop()
            self.deiconify()
            self.enable_menubar()

            if mask is not None:
                answer = self.ask_multiple_choice_question('Save edited mask as:', ('Overwrite the current object mask', 'Add as a new object', 'Do not save'))
                if answer == 0:
                    self.annotations[self.obj_id] = ObjectAnnotation(
                        np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8'),
                        self.annotations[self.obj_id].class_id
                    )
                elif answer == 1:
                    class_id = self.ask_class_id()
                    if class_id != -1:
                        self.annotations.insert(
                            self.obj_id + 1,
                            ObjectAnnotation(
                                np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8'),
                                class_id
                            )
                        )
        else:
            self.show_message('Please add an object by drawing a rectangle first', 'Guide')

        self.enable_callbacks()
        self.force_focus()


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
                    self.display()
            elif event.key in ['s', 'down']:
                if len(self.annotations) > 0:
                    self.obj_id = (self.obj_id - 1) % len(self.annotations)
                    self.display()
            elif event.key == 'ctrl+d':
                self.remove_current_object()
                self.display()
            elif event.key == 'ctrl+e':
                self.mask_touch_session()
                self.display()


    def on_mouse_press(self, event):
        super().on_mouse_press(event)
        p = self.get_image_coordinates(event)
        if event.key == 'shift' and event.button == 3:
            self.rect_ipr.start_dragging(p)


    def on_mouse_move(self, event):
        super().on_mouse_move(event)
        if self.rect_ipr.on_dragging:
            p = self.get_image_coordinates(event)
            self.rect_ipr.update(p)
            self.add_transient_patch(self.rect_ipr.rect.to_patch(
                linewidth=1,
                linestyle='--',
                edgecolor='b',
                facecolor='none'
            ))


    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        p = self.get_image_coordinates(event)
        if self.rect_ipr.on_dragging:
            self.rect_ipr.finish_dragging(p)
            class_id = self.ask_class_id()
            self.clear_transient_patch()
            if class_id != -1:
                mask = grabcut(self.img, cv2.GC_INIT_WITH_RECT, rect=self.rect_ipr.rect)
                if np.array_equal((mask == 1) + (mask == 3), np.zeros_like(mask)):
                    self.annotations.append(ObjectAnnotation(
                        self.rect_ipr.rect.to_mask(mask.shape),
                        class_id
                    ))
                else:
                    self.annotations.append(ObjectAnnotation(
                        np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8'),
                        class_id
                    ))
                self.obj_id = len(self.annotations) - 1
                self.display()

